import os
from collections import Counter

# ── CONFIG ───────────────────────────────────────────────────────────────────
DATA_DIR        = r'E:\EBM-NLP\ebm_nlp_2_00'
DOCS_DIR        = os.path.join(DATA_DIR, 'documents')
ANNOTATIONS_DIR = os.path.join(DATA_DIR, 'annotations', 'aggregated', 'starting_spans')

PICO_ELEMENTS = ['participants', 'interventions', 'outcomes']

# ── STEP 1: Load a single abstract ───────────────────────────────────────────
def load_document(pmid):
    text_path   = os.path.join(DOCS_DIR, f'{pmid}.txt')
    tokens_path = os.path.join(DOCS_DIR, f'{pmid}.tokens')

    with open(text_path,   'r', encoding='utf-8') as f:
        text = f.read().strip()
    with open(tokens_path, 'r', encoding='utf-8') as f:
        tokens = [t for t in f.read().strip().split('\n') if t.strip()]

    return text, tokens

# ── STEP 2: Load labels for one abstract ─────────────────────────────────────
def load_labels(pmid, element):
    for split in ['train', 'test']:
        label_path = os.path.join(ANNOTATIONS_DIR, element, split, f'{pmid}.AGGREGATED.ann')
        if os.path.exists(label_path):
            with open(label_path, 'r', encoding='utf-8') as f:
                labels = [int(x) for x in f.read().strip().split('\n') if x.strip()]
            return labels
    return None

# ── STEP 3: Extract labelled spans as text ───────────────────────────────────
def extract_spans(tokens, labels):
    spans        = []
    current_span = []

    for token, label in zip(tokens, labels):
        if label != 0:
            current_span.append(token)
        else:
            if current_span:
                spans.append(' '.join(current_span))
                current_span = []

    if current_span:
        spans.append(' '.join(current_span))

    return spans

# ── STEP 4: Load all documents ────────────────────────────────────────────────
def load_all_documents():
    all_pmids = [
        f.replace('.txt', '')
        for f in os.listdir(DOCS_DIR)
        if f.endswith('.txt')
    ]

    dataset = []
    for pmid in all_pmids:
        try:
            text, tokens = load_document(pmid)
        except Exception as e:
            print(f"  Skipping {pmid}: {e}")
            continue

        entry = {
            'pmid'  : pmid,
            'text'  : text,
            'tokens': tokens,
            'labels': {}
        }

        for element in PICO_ELEMENTS:
            labels = load_labels(pmid, element)
            if labels and len(labels) == len(tokens):
                entry['labels'][element]           = labels
                entry[f'{element}_spans']          = extract_spans(tokens, labels)

        dataset.append(entry)

    return dataset

# ── STEP 5: Explore the data ──────────────────────────────────────────────────
def explore(dataset):
    print(f"\n{'='*60}")
    print(f"  DATASET OVERVIEW")
    print(f"{'='*60}")
    print(f"  Total abstracts loaded : {len(dataset)}")

    lengths = [len(d['tokens']) for d in dataset]
    print(f"  Avg tokens per abstract: {sum(lengths) // len(lengths)}")
    print(f"  Min tokens             : {min(lengths)}")
    print(f"  Max tokens             : {max(lengths)}")

    print(f"\n{'='*60}")
    print(f"  LABEL COVERAGE (how often each field appears)")
    print(f"{'='*60}")
    for element in PICO_ELEMENTS:
        count = sum(1 for d in dataset if element in d['labels'])
        print(f"  {element:20s}: {count} / {len(dataset)} abstracts")

    print(f"\n{'='*60}")
    print(f"  MISSING LABELS CHECK")
    print(f"{'='*60}")
    for element in PICO_ELEMENTS:
        missing = sum(1 for d in dataset if element not in d['labels'])
        print(f"  {element:20s}: {missing} abstracts have NO labels")

    print(f"\n{'='*60}")
    print(f"  TOKEN-LABEL MISMATCH CHECK")
    print(f"{'='*60}")
    mismatches = 0
    for d in dataset:
        for element in PICO_ELEMENTS:
            if element in d['labels']:
                if len(d['labels'][element]) != len(d['tokens']):
                    mismatches += 1
    print(f"  Total mismatches found : {mismatches}")

    print(f"\n{'='*60}")
    print(f"  SAMPLE ABSTRACT (first one)")
    print(f"{'='*60}")
    sample = dataset[0]
    print(f"\n  PMID : {sample['pmid']}")
    print(f"\n  Text :\n  {sample['text'][:500]}...\n")

    for element in PICO_ELEMENTS:
        spans = sample.get(f'{element}_spans', [])
        print(f"  {element.upper()} spans:")
        if spans:
            for s in spans[:3]:
                print(f"    → {s}")
        else:
            print(f"    (none found)")
        print()

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Loading dataset... (this may take 30-60 seconds)")
    dataset = load_all_documents()

    if len(dataset) == 0:
        print("ERROR: No documents loaded. Check your DATA_DIR path.")
    else:
        explore(dataset)
        print("\nDone! Dataset loaded successfully.")