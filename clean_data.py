import os
import re
from collections import defaultdict

# ── CONFIG ───────────────────────────────────────────────────────────────────
DATA_DIR        = r'E:\EBM-NLP\ebm_nlp_2_00'
DOCS_DIR        = os.path.join(DATA_DIR, 'documents')
ANNOTATIONS_DIR = os.path.join(DATA_DIR, 'annotations', 'aggregated', 'starting_spans')
PICO_ELEMENTS   = ['participants', 'interventions', 'outcomes']
MIN_TOKENS      = 30  # abstracts shorter than this are too short to be useful

# ── LOAD FUNCTIONS  ──────────────────────────────────────────
def load_document(pmid):
    text_path   = os.path.join(DOCS_DIR, f'{pmid}.txt')
    tokens_path = os.path.join(DOCS_DIR, f'{pmid}.tokens')
    with open(text_path,   'r', encoding='utf-8') as f:
        text = f.read().strip()
    with open(tokens_path, 'r', encoding='utf-8') as f:
        tokens = [t for t in f.read().strip().split('\n') if t.strip()]
    return text, tokens

def load_labels(pmid, element):
    for split in ['train', 'test']:
        label_path = os.path.join(
            ANNOTATIONS_DIR, element, split, f'{pmid}.AGGREGATED.ann'
        )
        if os.path.exists(label_path):
            with open(label_path, 'r', encoding='utf-8') as f:
                labels = [int(x) for x in f.read().strip().split('\n') if x.strip()]
            return labels
    return None

def load_all_documents():
    all_pmids = [f.replace('.txt', '') for f in os.listdir(DOCS_DIR) if f.endswith('.txt')]
    dataset = []
    for pmid in all_pmids:
        try:
            text, tokens = load_document(pmid)
        except Exception as e:
            continue
        entry = {'pmid': pmid, 'text': text, 'tokens': tokens, 'labels': {}}
        for element in PICO_ELEMENTS:
            labels = load_labels(pmid, element)
            if labels and len(labels) == len(tokens):
                entry['labels'][element] = labels
        dataset.append(entry)
    return dataset

# ── CLEANING FUNCTIONS ────────────────────────────────────────────────────────

def clean_text(text):
    """Clean the raw abstract text."""
    # Remove square brackets e.g. [Triple therapy...]
    text = re.sub(r'\[', '', text)
    text = re.sub(r'\]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text

def clean_tokens(tokens):
    """Clean individual tokens."""
    cleaned = []
    for token in tokens:
        # Remove brackets
        token = token.replace('[', '').replace(']', '')
        # Strip whitespace
        token = token.strip()
        # Keep token only if it has actual content
        if token:
            cleaned.append(token)
    return cleaned

def is_valid_span(span):
    """Check if an extracted span is meaningful."""
    # Too short (single word or empty)
    if len(span.split()) < 2:
        return False
    # Only numbers or punctuation
    if re.match(r'^[\d\s\.\,\;\:]+$', span):
        return False
    return True

def extract_spans(tokens, labels):
    """Extract labelled spans and filter out noisy ones."""
    spans        = []
    current_span = []

    for token, label in zip(tokens, labels):
        if label != 0:
            current_span.append(token)
        else:
            if current_span:
                span = ' '.join(current_span)
                if is_valid_span(span):   # only keep meaningful spans
                    spans.append(span)
                current_span = []

    if current_span:
        span = ' '.join(current_span)
        if is_valid_span(span):
            spans.append(span)

    return spans

# ── MAIN CLEANING PIPELINE ────────────────────────────────────────────────────
def clean_dataset(dataset):
    cleaned      = []
    removed      = defaultdict(int)

    for entry in dataset:
        pmid   = entry['pmid']
        text   = entry['text']
        tokens = entry['tokens']
        labels = entry['labels']

        # ── FILTER 1: Remove abstracts that are too short ──────────────────
        if len(tokens) < MIN_TOKENS:
            removed['too_short'] += 1
            continue

        # ── FILTER 2: Remove abstracts missing ALL labels ──────────────────
        if len(labels) == 0:
            removed['no_labels'] += 1
            continue

        # ── CLEAN: text and tokens ─────────────────────────────────────────
        clean_text_val    = clean_text(text)
        clean_tokens_val  = clean_tokens(tokens)

        # ── EXTRACT: clean spans for each PIO element ──────────────────────
        spans = {}
        for element in PICO_ELEMENTS:
            if element in labels:
                # Re-align labels after token cleaning
                element_labels = labels[element][:len(clean_tokens_val)]
                spans[element] = extract_spans(clean_tokens_val, element_labels)
            else:
                spans[element] = []   # missing element — empty list

        # ── BUILD cleaned entry ────────────────────────────────────────────
        cleaned.append({
            'pmid'        : pmid,
            'text'        : clean_text_val,
            'tokens'      : clean_tokens_val,
            'labels'      : labels,
            'spans'       : spans
        })

    return cleaned, removed

# ── REPORT ────────────────────────────────────────────────────────────────────
def cleaning_report(original, cleaned, removed):
    print(f"\n{'='*60}")
    print(f"  CLEANING REPORT")
    print(f"{'='*60}")
    print(f"  Original abstracts     : {len(original)}")
    print(f"  Removed (too short)    : {removed['too_short']}")
    print(f"  Removed (no labels)    : {removed['no_labels']}")
    print(f"  Remaining after clean  : {len(cleaned)}")

    print(f"\n{'='*60}")
    print(f"  LABEL COVERAGE AFTER CLEANING")
    print(f"{'='*60}")
    for element in PICO_ELEMENTS:
        count   = sum(1 for d in cleaned if d['spans'][element])
        missing = len(cleaned) - count
        print(f"  {element:20s}: {count} have spans | {missing} missing")

    print(f"\n{'='*60}")
    print(f"  SAMPLE CLEANED ABSTRACT")
    print(f"{'='*60}")
    sample = cleaned[0]
    print(f"\n  PMID : {sample['pmid']}")
    print(f"\n  Text (cleaned):\n  {sample['text'][:400]}...\n")
    for element in PICO_ELEMENTS:
        print(f"  {element.upper()} spans:")
        spans = sample['spans'][element]
        if spans:
            for s in spans[:3]:
                print(f"    → {s}")
        else:
            print(f"    (none found)")
        print()

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Loading dataset...")
    dataset = load_all_documents()

    print("Cleaning dataset...")
    cleaned, removed = clean_dataset(dataset)

    cleaning_report(dataset, cleaned, removed)

    print("\nDone! Dataset is clean and ready.")
    