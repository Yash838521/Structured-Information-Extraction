import os
import re
import json
import csv
from collections import defaultdict

DATA_DIR        = os.environ.get('EBM_DATA_DIR', 'ebm_nlp_2_00')
DOCS_DIR        = os.path.join(DATA_DIR, 'documents')
ANNOTATIONS_DIR = os.path.join(DATA_DIR, 'annotations', 'aggregated', 'starting_spans')
OUTPUT_DIR      = os.environ.get('EBM_OUTPUT_DIR', 'cleaned_data')
PICO_ELEMENTS   = ['participants', 'interventions', 'outcomes']
MIN_TOKENS      = 10



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
    
    train_pmids = set()
    test_pmids  = set()

    train_ann_dir = os.path.join(ANNOTATIONS_DIR, 'participants', 'train')
    test_ann_dir  = os.path.join(ANNOTATIONS_DIR, 'participants', 'test', 'gold')

    if os.path.exists(train_ann_dir):
        for f in os.listdir(train_ann_dir):
            if f.endswith('.AGGREGATED.ann'):
                train_pmids.add(f.replace('.AGGREGATED.ann', ''))

    if os.path.exists(test_ann_dir):
        for f in os.listdir(test_ann_dir):
            if f.endswith('.AGGREGATED.ann'):
                test_pmids.add(f.replace('.AGGREGATED.ann', ''))

    print(f"  Found {len(train_pmids)} train PMIDs, {len(test_pmids)} test PMIDs")

    all_doc_pmids = set(
        f.replace('.txt', '') for f in os.listdir(DOCS_DIR) if f.endswith('.txt')  #included any PMIDs that have documents but aren't in participants
    )
    known = train_pmids | test_pmids
    unknown_pmids = all_doc_pmids - known

    all_pmids = (
        [(pmid, 'train') for pmid in train_pmids] +
        [(pmid, 'test')  for pmid in test_pmids] +
        [(pmid, None)    for pmid in unknown_pmids]
    )

    dataset = []
    for pmid, split in all_pmids:
        try:
            text, tokens = load_document(pmid)
        except Exception:
            continue

        entry = {'pmid': pmid, 'text': text, 'tokens': tokens,
                 'labels': {}, 'split': split}

        for element in PICO_ELEMENTS:
            labels, found_split = load_labels(pmid, element)
            if labels and len(labels) == len(tokens):
                entry['labels'][element] = labels
                # Update split if we didn't know it yet
                if entry['split'] is None and found_split:
                    entry['split'] = found_split

        dataset.append(entry)

    return dataset


#  CLEANING FUNCTIONS 

def clean_text(text):
    text = re.sub(r'[\[\]]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def clean_tokens_and_labels(tokens, labels_dict):
    cleaned_tokens = []
    cleaned_labels = {k: [] for k in labels_dict.keys()}

    for i, token in enumerate(tokens):
        clean_token = token.replace('[', '').replace(']', '').strip()

        if not clean_token:
            continue

        cleaned_tokens.append(clean_token)

        for element in labels_dict:
            cleaned_labels[element].append(labels_dict[element][i])

    return cleaned_tokens, cleaned_labels


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


def clean_dataset(dataset):
    cleaned = []
    removed = defaultdict(int)

    for entry in dataset:
        tokens = entry['tokens']
        labels = entry['labels']

        if len(tokens) < MIN_TOKENS:
            removed['too_short'] += 1
            continue

        if len(labels) == 0:
            removed['no_labels'] += 1
            continue

        clean_text_val   = clean_text(entry['text'])
        clean_tokens_val, clean_labels_val = clean_tokens_and_labels(tokens, labels)

        spans = {}
        for element in PICO_ELEMENTS:
            if element in labels:
                element_labels = clean_labels_val[element]
                spans[element] = extract_spans(clean_tokens_val, element_labels)
            else:
                spans[element] = []

        cleaned.append({
            'pmid'  : entry['pmid'],
            'split' : entry['split'],
            'text'  : clean_text_val,
            'tokens': clean_tokens_val,
            'labels': clean_labels_val,
            'spans' : spans
        })

    return cleaned, removed


# SAVE FUNCTIONS 

def save_dataset(cleaned):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_data = [d for d in cleaned if d['split'] == 'train']
    test_data  = [d for d in cleaned if d['split'] == 'test']

    full_path  = os.path.join(OUTPUT_DIR, 'cleaned_dataset.json')
    train_path = os.path.join(OUTPUT_DIR, 'train.json')
    test_path  = os.path.join(OUTPUT_DIR, 'test.json')

    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned, f, indent=2)
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2)
    with open(test_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2)

    
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


# FINAL REPORT 

def final_report(original, cleaned, removed, paths):
    full_path, train_path, test_path, csv_path = paths
    train_count = sum(1 for d in cleaned if d['split'] == 'train')
    test_count  = sum(1 for d in cleaned if d['split'] == 'test')


    print(f"  FINAL CLEANING SUMMARY")
    print(f"  Original abstracts       : {len(original)}")
    print(f"  Removed (too short < {MIN_TOKENS})  : {removed['too_short']}")
    print(f"  Removed (no labels)      : {removed['no_labels']}")
    print(f"  Final clean abstracts    : {len(cleaned)}")
    print(f"    Train set              : {train_count}")
    print(f"    Test set               : {test_count}")

    print(f"  LABEL COVERAGE")
    for element in PICO_ELEMENTS:
        count = sum(1 for d in cleaned if d['spans'][element])
        pct   = round(count / len(cleaned) * 100, 1)
        print(f"  {element:20s}: {count} / {len(cleaned)} ({pct}%)")

    print(f"  FILES SAVED")
    
    for label, path in [('Full dataset', full_path), ('Train set', train_path),
                         ('Test set', test_path), ('Summary CSV', csv_path)]:
        size = os.path.getsize(path) / (1024*1024)
        print(f"  {label:20s}: {path}  ({size:.1f} MB)")



#  MAIN
if __name__ == '__main__':
    print("Loading dataset")
    dataset = load_all_documents()

    print("Cleaning dataset")
    cleaned, removed = clean_dataset(dataset)

    print("Saving cleaned dataset")
    paths = save_dataset(cleaned)

    final_report(dataset, cleaned, removed, paths)
    print("\nCompleted")