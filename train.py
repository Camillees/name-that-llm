"""Fine-tune a small encoder to predict which LLM wrote a text.

The split is grouped by prompt_id so one prompt never sits in both train and
validation. That matches the LinearSVC notebook: the model has to use style,
not the shared topic.
"""

import gc
import inspect
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

ROOT = Path(__file__).resolve().parent
N_SPLITS = 5  # cut to 3 if the M4 run needs to be shorter
MODEL_CANDIDATES = (
    "microsoft/deberta-v3-small",
    # shortcut: DistilBERT only if DeBERTa attention fails on MPS,
    # switch back when torch implements the op.
    "distilbert-base-uncased",
)
MAX_LENGTH = 256
EPOCHS = 4
LR = 2e-5
BATCH_SIZE = 16
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
SEED = 42
BASELINE = 0.301


def assert_no_prompt_leakage(y, groups):
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    train_idx, valid_idx = next(cv.split(np.zeros(len(y)), y, groups))
    overlap = set(groups[train_idx]) & set(groups[valid_idx])
    if overlap:
        shown = ", ".join(sorted(overlap)[:8])
        print(f"prompt_id leakage across the split: {shown}", file=sys.stderr)
        sys.exit(1)


class EncodedTexts(torch.utils.data.Dataset):
    def __init__(self, tokenizer, texts, labels=None):
        encoded = tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )
        self.input_ids = encoded["input_ids"]
        self.attention_mask = encoded["attention_mask"]
        self.labels = None if labels is None else [int(v) for v in labels]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.argmax(logits, axis=-1)
    return {"balanced_accuracy": balanced_accuracy_score(labels, preds)}


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def smoke_test(model_name, num_labels, label2id, id2label, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label={str(i): name for i, name in id2label.items()},
        label2id=label2id,
    )
    model.to(device)
    model.train()
    batch = tokenizer(
        "style " * 180,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )
    batch = {
        key: value.to(device)
        for key, value in batch.items()
        if key in ("input_ids", "attention_mask")
    }
    batch["labels"] = torch.zeros(1, dtype=torch.long, device=device)
    loss = model(**batch).loss
    loss.backward()
    del model, loss, batch
    if device.type == "mps":
        torch.mps.empty_cache()
    return tokenizer


def load_tokenizer(num_labels, label2id, id2label, device):
    errors = []
    for name in MODEL_CANDIDATES:
        try:
            tokenizer = smoke_test(name, num_labels, label2id, id2label, device)
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
            print(f"skip {name}: {type(exc).__name__}: {exc}", flush=True)
            continue
        print(f"model: {name}", flush=True)
        return name, tokenizer
    raise RuntimeError("no model ran on this device\n" + "\n".join(errors))


def build_training_arguments(output_dir, epochs, do_eval):
    params = inspect.signature(TrainingArguments.__init__).parameters
    fields = set(getattr(TrainingArguments, "__dataclass_fields__", {}))
    accepted = set(params) | fields
    kw = {}

    def put(key, value):
        if key in accepted:
            kw[key] = value

    put("output_dir", output_dir)
    put("num_train_epochs", epochs)
    put("learning_rate", LR)
    put("per_device_train_batch_size", BATCH_SIZE)
    put("per_device_eval_batch_size", BATCH_SIZE)
    put("weight_decay", WEIGHT_DECAY)
    # transformers 5: a warmup_steps float in [0, 1) is a ratio of total steps.
    put("warmup_steps", WARMUP_RATIO)
    put("logging_strategy", "epoch")
    put("report_to", "none")
    put("seed", SEED)
    put("data_seed", SEED)
    put("dataloader_pin_memory", False)
    put("dataloader_num_workers", 0)
    put("fp16", False)
    put("bf16", False)
    put("use_cpu", False)
    put("overwrite_output_dir", True)
    put("save_total_limit", 1)
    eval_value = "epoch" if do_eval else "no"
    if "eval_strategy" in accepted:
        put("eval_strategy", eval_value)
    else:
        put("evaluation_strategy", eval_value)
    put("save_strategy", "epoch" if do_eval else "no")
    if do_eval:
        put("load_best_model_at_end", True)
        put("metric_for_best_model", "balanced_accuracy")
        put("greater_is_better", True)
    return TrainingArguments(**kw)


def make_trainer(
    model_name, tokenizer, train_ds, eval_ds, epochs, output_dir, label2id, id2label
):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        id2label={str(i): name for i, name in id2label.items()},
        label2id=label2id,
    )
    do_eval = eval_ds is not None
    trainer_kw = {
        "model": model,
        "args": build_training_arguments(output_dir, epochs, do_eval),
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "compute_metrics": compute_metrics if do_eval else None,
    }
    signature = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in signature:
        trainer_kw["processing_class"] = tokenizer
    else:
        trainer_kw["tokenizer"] = tokenizer
    return Trainer(**trainer_kw)


def best_epoch_score(trainer):
    best_score = None
    best_epoch = None
    for row in trainer.state.log_history:
        score = row.get("eval_balanced_accuracy")
        if score is None:
            continue
        if best_score is None or score > best_score:
            best_score = float(score)
            best_epoch = int(round(row["epoch"]))
    if best_score is None:
        raise RuntimeError("validation never reported balanced accuracy")
    return best_epoch, best_score


def main():
    set_seed(SEED)
    train = pd.read_csv(ROOT / "train.csv")
    label_names = sorted(train["model"].unique())
    label2id = {name: i for i, name in enumerate(label_names)}
    id2label = {i: name for name, i in label2id.items()}
    texts = train["text"].fillna("").tolist()
    y = train["model"].map(label2id).to_numpy()
    groups = train["prompt_id"].to_numpy()

    assert_no_prompt_leakage(y, groups)
    print(f"leakage check passed ({len(label_names)} labels, {len(train)} rows)", flush=True)

    device = pick_device()
    print(f"device: {device}", flush=True)
    model_name, tokenizer = load_tokenizer(len(label_names), label2id, id2label, device)

    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    fold_scores = []
    best_epochs = []
    for fold, (train_idx, valid_idx) in enumerate(
        cv.split(np.zeros(len(y)), y, groups), start=1
    ):
        overlap = set(groups[train_idx]) & set(groups[valid_idx])
        if overlap:
            print(f"prompt_id leakage in fold {fold}", file=sys.stderr)
            sys.exit(1)
        out_dir = ROOT / "checkpoints" / f"fold-{fold}"
        shutil.rmtree(out_dir, ignore_errors=True)
        trainer = make_trainer(
            model_name,
            tokenizer,
            EncodedTexts(tokenizer, [texts[i] for i in train_idx], y[train_idx]),
            EncodedTexts(tokenizer, [texts[i] for i in valid_idx], y[valid_idx]),
            EPOCHS,
            str(out_dir),
            label2id,
            id2label,
        )
        param_device = next(trainer.model.parameters()).device
        print(f"fold {fold} parameters on {param_device}", flush=True)
        trainer.train()
        epoch, score = best_epoch_score(trainer)
        fold_scores.append(score)
        best_epochs.append(epoch)
        print(f"Fold {fold}: {score:.4f} (best epoch {epoch})", flush=True)
        del trainer
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
        shutil.rmtree(out_dir, ignore_errors=True)

    mean = float(np.mean(fold_scores))
    std = float(np.std(fold_scores))
    print(f"\nMean CV score: {mean:.4f}", flush=True)
    print(f"Standard deviation: {std:.4f}", flush=True)
    print(f"LinearSVC baseline: {BASELINE:.3f}", flush=True)
    print(f"Difference vs baseline: {mean - BASELINE:+.4f}", flush=True)

    final_epochs = max(1, int(round(float(np.median(best_epochs)))))
    print(f"Final fit epochs: {final_epochs}", flush=True)
    final_dir = ROOT / "checkpoints" / "final"
    shutil.rmtree(final_dir, ignore_errors=True)
    trainer = make_trainer(
        model_name,
        tokenizer,
        EncodedTexts(tokenizer, texts, y),
        None,
        final_epochs,
        str(final_dir),
        label2id,
        id2label,
    )
    trainer.train()

    test = pd.read_csv(ROOT / "test.csv")
    predicted = trainer.predict(EncodedTexts(tokenizer, test["text"].fillna("").tolist()))
    logits = predicted.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    pred_ids = np.argmax(logits, axis=-1)
    out = pd.DataFrame(
        {"id": test["id"], "model": [label_names[int(i)] for i in pred_ids]}
    )
    out_path = ROOT / "predictions.csv"
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False), flush=True)
    print(f"wrote {out_path}", flush=True)
    del trainer
    shutil.rmtree(final_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
