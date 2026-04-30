#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        Trainer,
        TrainingArguments,
        set_seed,
    )
except Exception:
    raise ImportError("Please install transformers and torch first: pip install transformers torch")


ELEMENTS = ["participants", "interventions", "outcomes"]

LABEL_LIST = [
    "O",
    "B-PARTICIPANTS", "I-PARTICIPANTS",
    "B-INTERVENTIONS", "I-INTERVENTIONS",
    "B-OUTCOMES", "I-OUTCOMES",
]
LABEL2ID = {lab: i for i, lab in enumerate(LABEL_LIST)}
ID2LABEL = {i: lab for lab, i in LABEL2ID.items()}


class Example:
    def __init__(self, pmid, tokens, text, split, word_labels):
        self.pmid = pmid
        self.tokens = tokens
        self.text = text
        self.split = split
        self.word_labels = word_labels


def load_json(path: str | Path) -> List[Dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def normalize_ws(text: str) -> str:
    import re
    return re.sub(r"\s+", " ", str(text)).strip()


def dedupe_keep_order(items: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        item = normalize_ws(item)
        if item and item.lower() not in seen:
            seen.add(item.lower())
            out.append(item)
    return out


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)


# In[2]:


def binary_to_bio(binary_mask, prefix):
    out = []
    prev = 0
    for x in binary_mask:
        x = int(x)
        if x == 1:
            out.append(f"I-{prefix}" if prev == 1 else f"B-{prefix}")
        else:
            out.append("O")
        prev = x
    return out


def merge_label_sequences(seqs):
    if not seqs:
        return []
    merged = ["O"] * len(seqs[0])
    for seq in seqs:
        for i, lab in enumerate(seq):
            if lab != "O":
                merged[i] = lab
    return merged


def build_word_labels(item):
    tokens = item.get("tokens", [])
    labels = item.get("labels", {}) or {}
    n = len(tokens)
    if n == 0:
        return []

    part_bio = binary_to_bio(labels.get("participants", [0] * n), "PARTICIPANTS")
    int_bio = binary_to_bio(labels.get("interventions", [0] * n), "INTERVENTIONS")
    out_bio = binary_to_bio(labels.get("outcomes", [0] * n), "OUTCOMES")
    merged = merge_label_sequences([part_bio, int_bio, out_bio])
    return [LABEL2ID[x] for x in merged]


def build_examples(data, split_filter: Optional[str] = None):
    out = []
    for item in data:
        split = str(item.get("split", "")).strip()
        if split_filter and split != split_filter:
            continue
        tokens = list(item.get("tokens", []) or [])
        if not tokens:
            continue
        out.append(
            Example(
                str(item["pmid"]),
                tokens,
                str(item["text"]),
                split,
                build_word_labels(item),
            )
        )
    return out


# In[3]:


class TokenClassificationDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=256):
        self.examples = list(examples)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.features = [self._encode(ex) for ex in self.examples]

    def _encode(self, ex):
        enc = self.tokenizer(
            ex.tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
        )
        word_ids = enc.word_ids()
        labels = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                labels.append(-100)
            elif word_id != prev_word_id:
                labels.append(ex.word_labels[word_id])
            else:
                labels.append(-100)
            prev_word_id = word_id
        enc["labels"] = labels
        return enc

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        item = self.features[idx]
        return {k: torch.tensor(v, dtype=torch.long) for k, v in item.items()}


# In[4]:


def label_id_to_field(label_id):
    label = ID2LABEL[int(label_id)]
    if label == "O":
        return None
    if label.endswith("PARTICIPANTS"):
        return "participants"
    if label.endswith("INTERVENTIONS"):
        return "interventions"
    if label.endswith("OUTCOMES"):
        return "outcomes"
    return None


def decode_single_prediction(logits, ex, feature):
    pred_ids = np.argmax(logits, axis=-1)
    word_ids = feature.word_ids()
    word_pred_labels = {}
    for token_idx, word_id in enumerate(word_ids):
        if word_id is not None and word_id not in word_pred_labels:
            word_pred_labels[word_id] = int(pred_ids[token_idx])

    ordered_pred_ids = [word_pred_labels.get(i, LABEL2ID["O"]) for i in range(len(ex.tokens))]
    spans = {e: [] for e in ELEMENTS}

    current_field = None
    current_tokens = []

    def flush():
        nonlocal current_field, current_tokens
        if current_field and current_tokens:
            spans[current_field].append(" ".join(current_tokens))
        current_field = None
        current_tokens = []

    for tok, lab_id in zip(ex.tokens, ordered_pred_ids):
        label = ID2LABEL[lab_id]
        if label == "O":
            flush()
            continue
        field = label_id_to_field(lab_id)
        if field is None:
            flush()
            continue
        if label.startswith("B-"):
            flush()
            current_field = field
            current_tokens = [tok]
        else:
            if current_field == field:
                current_tokens.append(tok)
            else:
                flush()
                current_field = field
                current_tokens = [tok]
    flush()

    return {k: dedupe_keep_order(v) for k, v in spans.items()}


# In[5]:


from collections import Counter

def normalize_span_text(text: str) -> str:
    return normalize_ws(str(text)).lower()

def compute_exact_span_scores(predicted_spans: Sequence[str], gold_spans: Sequence[str]) -> Tuple[float, float, float]:
    pred_counter = Counter(
        normalize_span_text(span) for span in predicted_spans if normalize_span_text(span)
    )
    gold_counter = Counter(
        normalize_span_text(span) for span in gold_spans if normalize_span_text(span)
    )

    pred_total = sum(pred_counter.values())
    gold_total = sum(gold_counter.values())
    tp = sum((pred_counter & gold_counter).values())

    if pred_total == 0 and gold_total == 0:
        return 1.0, 1.0, 1.0
    if pred_total == 0:
        return 0.0, 0.0, 0.0
    if gold_total == 0:
        return 0.0, 0.0, 0.0

    precision = tp / pred_total
    recall = tp / gold_total
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def evaluate_pipeline(pipeline, test_data: List[Dict], save_predictions: bool = True):
    results = {e: {"precisions": [], "recalls": [], "f1s": [], "coverage": 0} for e in ELEMENTS}
    all_predictions = []

    for idx, doc in enumerate(test_data):
        if idx % 20 == 0:
            print(f"      processing {idx+1}/{len(test_data)}...")

        preds = pipeline.extract(doc)

        if save_predictions:
            all_predictions.append({
                "pmid": doc["pmid"],
                "predictions": preds,
                "gold": doc.get("spans", {}),
            })

        for elem in ELEMENTS:
            gold = doc.get("spans", {}).get(elem, []) or []
            pred = preds.get(elem, []) or []

            p, r, f = compute_exact_span_scores(pred, gold)
            results[elem]["precisions"].append(p)
            results[elem]["recalls"].append(r)
            results[elem]["f1s"].append(f)

            if len(pred) > 0:
                results[elem]["coverage"] += 1

    summary = {}
    for elem in ELEMENTS:
        n = len(results[elem]["f1s"])
        if n == 0:
            summary[elem] = {"precision": 0, "recall": 0, "f1": 0, "coverage": 0, "n": 0}
        else:
            summary[elem] = {
                "precision": round(float(np.mean(results[elem]["precisions"])), 4),
                "recall": round(float(np.mean(results[elem]["recalls"])), 4),
                "f1": round(float(np.mean(results[elem]["f1s"])), 4),
                "coverage": round(results[elem]["coverage"] / n, 4),
                "n": n,
            }

    return summary, all_predictions


# In[6]:


def print_results_table(all_results: Dict[str, Dict]):
    print(f"\n{'='*88}")
    print("  EVALUATION RESULTS")
    print(f"{'='*88}")
    print(f"  {'Pipeline':<22s} {'Element':<16s} {'Prec':>7s} {'Rec':>7s} {'F1':>7s} {'Cov%':>7s}  {'N':>4s}")
    print(f"  {'-'*82}")

    for pname, summary in all_results.items():
        for i, elem in enumerate(ELEMENTS):
            s = summary[elem]
            label = pname if i == 0 else ""
            print(
                f"  {label:<22s} {elem:<16s} {s['precision']:>7.3f} "
                f"{s['recall']:>7.3f} {s['f1']:>7.3f} {s['coverage']*100:>6.1f}%  {s['n']:>4d}"
            )
        print(f"  {'-'*82}")

    print(f"\n  {'Pipeline':<22s} {'Macro-F1':>9s} {'Macro-Prec':>11s} {'Macro-Rec':>10s}")
    print(f"  {'-'*56}")
    for pname, summary in all_results.items():
        mf = np.mean([summary[e]["f1"] for e in ELEMENTS])
        mp = np.mean([summary[e]["precision"] for e in ELEMENTS])
        mr = np.mean([summary[e]["recall"] for e in ELEMENTS])
        print(f"  {pname:<22s} {mf:>9.3f} {mp:>11.3f} {mr:>10.3f}")


class TokenClassificationPipeline:
    def __init__(
        self,
        model_name="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        model_dir="./biomedbert_token_classifier_model",
        max_length=256,
        learning_rate=2e-5,
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        seed=123,
        dev_ratio=0.1,
    ):
        self.name = "Token-Classification-BiomedBERT"
        self.model_name = model_name
        self.model_dir = Path(model_dir)
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.weight_decay = weight_decay
        self.seed = seed
        self.dev_ratio = dev_ratio
        self.tokenizer = None
        self.model = None

    def train(self, train_data):
        set_all_seeds(self.seed)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        examples = build_examples(train_data, split_filter="train")
        if not examples:
            raise ValueError("No training examples found.")

        random.Random(self.seed).shuffle(examples)
        n_dev = max(1, int(len(examples) * self.dev_ratio)) if len(examples) >= 10 else 0
        dev_examples = examples[:n_dev] if n_dev > 0 else []
        fit_examples = examples[n_dev:] if n_dev > 0 else examples

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=len(LABEL_LIST),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )

        train_dataset = TokenClassificationDataset(fit_examples, self.tokenizer, max_length=self.max_length)
        eval_dataset = TokenClassificationDataset(dev_examples, self.tokenizer, max_length=self.max_length) if dev_examples else None
        data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)

        args = TrainingArguments(
            output_dir=str(self.model_dir),
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            num_train_epochs=self.num_train_epochs,
            weight_decay=self.weight_decay,
            logging_steps=20,
            save_strategy="epoch",
            eval_strategy="epoch" if eval_dataset is not None else "no",
            report_to="none",
            load_best_model_at_end=True if eval_dataset is not None else False,
            metric_for_best_model="eval_loss" if eval_dataset is not None else None,
            greater_is_better=False,
            seed=self.seed,
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
            data_collator=data_collator,
        )
        trainer.train()
        trainer.save_model(str(self.model_dir))
        self.tokenizer.save_pretrained(str(self.model_dir))

    def _lazy_load(self):
        if self.model is not None and self.tokenizer is not None:
            return True
        if self.model_dir.exists():
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
            self.model = AutoModelForTokenClassification.from_pretrained(str(self.model_dir))
            return True
        return False

    def extract(self, doc):
        if not self._lazy_load():
            return {e: [] for e in ELEMENTS}

        ex = Example(
            str(doc["pmid"]),
            list(doc.get("tokens", []) or []),
            str(doc["text"]),
            str(doc.get("split", "")),
            [0] * len(doc.get("tokens", []) or []),
        )
        dataset = TokenClassificationDataset([ex], self.tokenizer, max_length=self.max_length)
        data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)
        args = TrainingArguments(output_dir="./tmp_predict_output", per_device_eval_batch_size=1, report_to="none")
        trainer = Trainer(model=self.model, args=args, processing_class=self.tokenizer, data_collator=data_collator)
        preds = trainer.predict(dataset)
        return decode_single_prediction(preds.predictions[0], ex, dataset.features[0])


# In[7]:


def main():
    train = load_json("../../cleaned_data/train.json")
    test = load_json("../../cleaned_data/test.json")

    pipe = TokenClassificationPipeline()
    print("Training token classification pipeline...")
    pipe.train(train)

    print("Evaluating on test set...")
    summary, predictions = evaluate_pipeline(pipe, test, save_predictions=True)

    all_results = {pipe.name: summary}
    print_results_table(all_results)

    output = {
        "summaries": all_results,
        "predictions": {
            pipe.name: predictions
        }
    }

    out_path = Path("Token-Classification-BiomedBERT_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nAll results saved to: {out_path}")


if __name__ == "__main__":
    main()



