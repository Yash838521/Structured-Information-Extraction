

import os
import re
from collections import defaultdict

DATA_DIR        = os.environ.get('EBM_DATA_DIR', 'ebm_nlp_2_00')
DOCS_DIR        = os.path.join(DATA_DIR, 'documents')
ANNOTATIONS_DIR = os.path.join(DATA_DIR, 'annotations', 'aggregated', 'starting_spans')
PICO_ELEMENTS   = ['participants', 'interventions', 'outcomes']

MIN_TOKENS = 10


#LOAD FUNCTIONS 

def load_document(pmid):
    
    text_path   = os.path.join(DOCS_DIR, f'{pmid}.txt')
    tokens_path = os.path.join(DOCS_DIR, f'{pmid}.tokens')
    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    with open(tokens_path, 'r', encoding='utf-8') as f:
        tokens = [t for t in f.read().strip().split('\n') if t.strip()]
    return text, tokens


def load_labels(pmid, element):
    
    
    train_path = os.path.join(
        ANNOTATIONS_DIR, element, 'train', f'{pmid}.AGGREGATED.ann'
    )
    if os.path.exists(train_path):
        with open(train_path, 'r', encoding='utf-8') as f:
            return [int(x) for x in f.read().strip().split('\n') if x.strip()], 'train'

    test_path = os.path.join(
        ANNOTATIONS_DIR, element, 'test', 'gold', f'{pmid}.AGGREGATED.ann'
    )
    if os.path.exists(test_path):
        with open(test_path, 'r', encoding='utf-8') as f:
            return [int(x) for x in f.read().strip().split('\n') if x.strip()], 'test'

    return None, None


def load_all_documents():
    
    all_pmids = [f.replace('.txt', '') for f in os.listdir(DOCS_DIR) if f.endswith('.txt')]
    dataset = []

    for pmid in all_pmids:
        try:
            text, tokens = load_document(pmid)
        except Exception:
            continue

        entry = {'pmid': pmid, 'text': text, 'tokens': tokens,
                 'labels': {}, 'split': None}

        for element in PICO_ELEMENTS:
            labels, split = load_labels(pmid, element)
            if labels and len(labels) == len(tokens):
                entry['labels'][element] = labels
                if entry['split'] is None:
                    entry['split'] = split

        dataset.append(entry)
    return dataset


# CLEANING FUNCTIONS 

def clean_text(text):
    
    text = re.sub(r'[\[\]]', '', text)     
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
    if re.match(r'^[\d\s\.\,\;\:\-\(\)%]+$', span):
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


# MAIN CLEANING PIPELINE 

def clean_dataset(dataset):
    
    cleaned = []
    removed = defaultdict(int)

    for entry in dataset:
        tokens = entry['tokens']
        labels = entry['labels']

        # FILTER 1:  
        if len(tokens) < MIN_TOKENS:  #can't extract meaningful PICO info
            removed['too_short'] += 1
            continue

        # FILTER 2:  
        if len(labels) == 0:   # no annotations at all
            removed['no_labels'] += 1
            continue

        
        clean_text_val   = clean_text(entry['text'])
        clean_tokens_val = clean_tokens(tokens)

        
        spans = {}                               #readable spans for each PIO element
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


# REPORTING

def cleaning_report(original, cleaned, removed):
    print(f"\n{'='*60}")
    print(f"  CLEANING REPORT")
    print(f"{'='*60}")
    print(f"  Original abstracts     : {len(original)}")
    print(f"  Removed (too short)    : {removed['too_short']}  (< {MIN_TOKENS} tokens)")
    print(f"  Removed (no labels)    : {removed['no_labels']}")
    print(f"  Remaining after clean  : {len(cleaned)}")

    train_n = sum(1 for d in cleaned if d['split'] == 'train')
    test_n  = sum(1 for d in cleaned if d['split'] == 'test')
    print(f"    Train                : {train_n}")
    print(f"    Test                 : {test_n}")

    
    print(f"  LABEL COVERAGE AFTER CLEANING")
    for element in PICO_ELEMENTS:
        count   = sum(1 for d in cleaned if d['spans'][element])
        missing = len(cleaned) - count
        print(f"  {element:20s}: {count} have spans | {missing} missing")

    print(f"  SAMPLE CLEANED ABSTRACT")
    
    sample = next((d for d in cleaned if all(d['spans'][e] for e in PICO_ELEMENTS)), cleaned[0])
    print(f"\n  PMID  : {sample['pmid']}")
    print(f"  Split : {sample['split']}")
    print(f"\n  Text (cleaned):\n  {sample['text'][:400]}...\n")
    for element in PICO_ELEMENTS:
        print(f"  {element.upper()} spans:")
        spans = sample['spans'][element]
        if spans:
            for s in spans[:3]:
                print(f"    → {s}")
        else:
            print(f"    (none)")
        print()


# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Loading dataset")
    dataset = load_all_documents()

    print("Cleaning dataset")
    cleaned, removed = clean_dataset(dataset)

    cleaning_report(dataset, cleaned, removed)
    print("\n Dataset is clean and ready.")