import os
import re
import json
from collections import defaultdict

# ── CONFIG ───────────────────────────────────────────────────────────────────
DATA_DIR        = r'E:\EBM-NLP\ebm_nlp_2_00'
DOCS_DIR        = os.path.join(DATA_DIR, 'documents')
ANNOTATIONS_DIR = os.path.join(DATA_DIR, 'annotations', 'aggregated', 'starting_spans')
OUTPUT_DIR      = r'E:\EBM-NLP\cleaned_data'
PICO_ELEMENTS   = ['participants', 'interventions', 'outcomes']
MIN_TOKENS      = 30

# ── LOAD FUNCTIONS ────────────────────────────────────────────────────────────
def load_document(pmid):
    text_path   = os.path.join(DOCS_DIR, f'{pmid}.txt')
    tokens_path = os.path.join(DOCS_DIR, f'{pmid}.tokens')
    with open(text_path,   'r', encoding='utf-8') as f:
        text = f.read().strip()
    with open(tokens_path, 'r', encoding='utf-8') as f:
        tokens = [t for t in f.read().strip().split('\n') if t.strip()]
    return text, tokens

def load_labels(pmid, element):
    # Check train
    train_path = os.path.join(ANNOTATIONS_DIR, element, 'train', f'{pmid}.AGGREGATED.ann')
    if os.path.exists(train_path):
        with open(train_path, 'r', encoding='utf-8') as f:
            labels = [int(x) for x in f.read().strip().split('\n') if x.strip()]
        return labels, 'train'

    # Check test/gold
    test_path = os.path.join(ANNOTATIONS_DIR, element, 'test', 'gold', f'{pmid}.AGGREGATED.ann')
    if os.path.exists(test_path):
        with open(test_path, 'r', encoding='utf-8') as f:
            labels = [int(x) for x in f.read().strip().split('\n') if x.strip()]
        return labels, 'test'

    return None, None

def load_all_documents():
    # First, collect all PMIDs from train and test/gold separately
    train_pmids = set()
    test_pmids  = set()

    train_ann_dir = os.path.join(ANNOTATIONS_DIR, 'participants', 'train')
    test_ann_dir  = os.path.join(ANNOTATIONS_DIR, 'participants', 'test', 'gold')

    for f in os.listdir(train_ann_dir):
        if f.endswith('.AGGREGATED.ann'):
            train_pmids.add(f.replace('.AGGREGATED.ann', ''))

    for f in os.listdir(test_ann_dir):
        if f.endswith('.AGGREGATED.ann'):
            test_pmids.add(f.replace('.AGGREGATED.ann', ''))

    print(f"  Found {len(train_pmids)} train PMIDs")
    print(f"  Found {len(test_pmids)} test PMIDs")

    all_pmids = [(pmid, 'train') for pmid in train_pmids] + \
                [(pmid, 'test')  for pmid in test_pmids]

    dataset = []
    for pmid, split in all_pmids:
        try:
            text, tokens = load_document(pmid)
        except:
            continue

        entry = {'pmid': pmid, 'text': text, 'tokens': tokens, 'labels': {}, 'split': split}

        for element in PICO_ELEMENTS:
            labels, _ = load_labels(pmid, element)
            if labels and len(labels) == len(tokens):
                entry['labels'][element] = labels

        dataset.append(entry)

    return dataset

# ── CLEANING FUNCTIONS ────────────────────────────────────────────────────────
def clean_text(text):
    text = re.sub(r'\[', '', text)
    text = re.sub(r'\]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def clean_tokens(tokens):
    cleaned = []
    for token in tokens:
        token = token.replace('[', '').replace(']', '').strip()
        if token:
            cleaned.append(token)
    return cleaned

def is_valid_span(span):
    if len(span.split()) < 2:
        return False
    if re.match(r'^[\d\s\.\,\;\:]+$', span):
        return False
    return True

def extract_spans(tokens, labels):
    spans        = []
    current_span = []
    for token, label in zip(tokens, labels):
        if label != 0:
            current_span.append(token)
        else:
            if current_span:
                span = ' '.join(current_span)
                if is_valid_span(span):
                    spans.append(span)
                current_span = []
    if current_span:
        span = ' '.join(current_span)
        if is_valid_span(span):
            spans.append(span)
    return spans

# ── MAIN CLEANING PIPELINE ────────────────────────────────────────────────────
def clean_dataset(dataset):
    cleaned = []
    removed = defaultdict(int)

    for entry in dataset:
        tokens = entry['tokens']
        labels = entry['labels']

        # Filter 1: too short
        if len(tokens) < MIN_TOKENS:
            removed['too_short'] += 1
            continue

        # Filter 2: no labels at all
        if len(labels) == 0:
            removed['no_labels'] += 1
            continue

        clean_text_val   = clean_text(entry['text'])
        clean_tokens_val = clean_tokens(tokens)

        spans = {}
        for element in PICO_ELEMENTS:
            if element in labels:
                element_labels = labels[element][:len(clean_tokens_val)]
                spans[element] = extract_spans(clean_tokens_val, element_labels)
            else:
                spans[element] = []

        cleaned.append({
            'pmid'  : entry['pmid'],
            'split' : entry['split'],
            'text'  : clean_text_val,
            'tokens': clean_tokens_val,
            'labels': {k: v[:len(clean_tokens_val)] for k, v in labels.items()},
            'spans' : spans
        })

    return cleaned, removed

# ── SAVE FUNCTIONS ────────────────────────────────────────────────────────────
def save_dataset(cleaned):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Split into train and test
    train_data = [d for d in cleaned if d['split'] == 'train']
    test_data  = [d for d in cleaned if d['split'] == 'test']

    # Save full cleaned dataset as JSON
    full_path  = os.path.join(OUTPUT_DIR, 'cleaned_dataset.json')
    train_path = os.path.join(OUTPUT_DIR, 'train.json')
    test_path  = os.path.join(OUTPUT_DIR, 'test.json')

    with open(full_path,  'w', encoding='utf-8') as f:
        json.dump(cleaned,    f, indent=2)
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2)
    with open(test_path,  'w', encoding='utf-8') as f:
        json.dump(test_data,  f, indent=2)

    # Save a human readable summary CSV
    import csv
    csv_path = os.path.join(OUTPUT_DIR, 'cleaned_summary.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'pmid', 'split', 'num_tokens',
            'participants_spans', 'interventions_spans', 'outcomes_spans',
            'text_preview'
        ])
        for d in cleaned:
            writer.writerow([
                d['pmid'],
                d['split'],
                len(d['tokens']),
                ' | '.join(d['spans']['participants']),
                ' | '.join(d['spans']['interventions']),
                ' | '.join(d['spans']['outcomes']),
                d['text'][:150]
            ])

    return full_path, train_path, test_path, csv_path

# ── FINAL REPORT ──────────────────────────────────────────────────────────────
def final_report(original, cleaned, removed, paths):
    full_path, train_path, test_path, csv_path = paths
    train_count = sum(1 for d in cleaned if d['split'] == 'train')
    test_count  = sum(1 for d in cleaned if d['split'] == 'test')

    print(f"\n{'='*60}")
    print(f"  FINAL CLEANING SUMMARY")
    print(f"{'='*60}")
    print(f"  Original abstracts       : {len(original)}")
    print(f"  Removed (too short)      : {removed['too_short']}")
    print(f"  Removed (no labels)      : {removed['no_labels']}")
    print(f"  Final clean abstracts    : {len(cleaned)}")
    print(f"  Train set                : {train_count}")
    print(f"  Test set                 : {test_count}")

    print(f"\n{'='*60}")
    print(f"  LABEL COVERAGE")
    print(f"{'='*60}")
    for element in PICO_ELEMENTS:
        count   = sum(1 for d in cleaned if d['spans'][element])
        pct     = round(count / len(cleaned) * 100, 1)
        print(f"  {element:20s}: {count} / {len(cleaned)} ({pct}%)")

    print(f"\n{'='*60}")
    print(f"  FILES SAVED")
    print(f"{'='*60}")
    print(f"  Full dataset (JSON) : {full_path}")
    print(f"  Train set    (JSON) : {train_path}")
    print(f"  Test set     (JSON) : {test_path}")
    print(f"  Summary      (CSV)  : {csv_path}")
    print(f"\n  Share the cleaned_data folder with your teammates!")

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Loading dataset...")
    dataset = load_all_documents()

    print("Cleaning dataset...")
    cleaned, removed = clean_dataset(dataset)

    print("Saving cleaned dataset...")
    paths = save_dataset(cleaned)

    final_report(dataset, cleaned, removed, paths)
    print("\nAll done!")