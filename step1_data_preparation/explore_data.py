

import os
from collections import Counter

DATA_DIR        = os.environ.get('EBM_DATA_DIR', 'ebm_nlp_2_00')
DOCS_DIR        = os.path.join(DATA_DIR, 'documents')
ANNOTATIONS_DIR = os.path.join(DATA_DIR, 'annotations', 'aggregated', 'starting_spans')
PICO_ELEMENTS   = ['participants', 'interventions', 'outcomes']


#  LOAD FUNCTIONS 

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
            labels = [int(x) for x in f.read().strip().split('\n') if x.strip()]
        return labels, 'train'

    
    test_path = os.path.join(
        ANNOTATIONS_DIR, element, 'test', 'gold', f'{pmid}.AGGREGATED.ann'
    )
    if os.path.exists(test_path):
        with open(test_path, 'r', encoding='utf-8') as f:
            labels = [int(x) for x in f.read().strip().split('\n') if x.strip()]
        return labels, 'test'

    return None, None


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


def load_all_documents():
    all_pmids = [
        f.replace('.txt', '')
        for f in os.listdir(DOCS_DIR) if f.endswith('.txt')
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
            'labels': {},
            'split' : None
        }

        for element in PICO_ELEMENTS:
            labels, split = load_labels(pmid, element)
            if labels and len(labels) == len(tokens):
                entry['labels'][element] = labels
                if entry['split'] is None:
                    entry['split'] = split

        dataset.append(entry)

    return dataset


# EXPLORATION REPORT 

def explore(dataset):

    print(f" 1. DATASET OVERVIEW")
    print(f"  Total abstracts loaded : {len(dataset)}")

    train_count = sum(1 for d in dataset if d['split'] == 'train')
    test_count  = sum(1 for d in dataset if d['split'] == 'test')
    none_count  = sum(1 for d in dataset if d['split'] is None)
    print(f"  Train split            : {train_count}")
    print(f"  Test split             : {test_count}")
    print(f"  No annotations at all  : {none_count}")


    lengths = [len(d['tokens']) for d in dataset]
    lengths_sorted = sorted(lengths)
    n = len(lengths)


    print(f"  2. TOKEN LENGTH STATISTICS")
   
    print(f"  Min tokens             : {min(lengths)}")
    print(f"  5th percentile         : {lengths_sorted[int(n*0.05)]}")
    print(f"  Median                 : {lengths_sorted[n//2]}")
    print(f"  Mean                   : {sum(lengths)//n}")
    print(f"  95th percentile        : {lengths_sorted[int(n*0.95)]}")
    print(f"  Max tokens             : {max(lengths)}")

    print(f"\n  Documents removed at each MIN_TOKENS threshold:")
    for thresh in [5, 10, 15, 20, 30]:
        removed = sum(1 for l in lengths if l < thresh)
        print(f"    MIN_TOKENS={thresh:2d}  →  removes {removed:3d} docs ({removed/n*100:.1f}%)")


    print(f"  3. LABEL COVERAGE (how many abstracts have each element)")
    for element in PICO_ELEMENTS:
        count   = sum(1 for d in dataset if element in d['labels'])
        missing = n - count
        print(f"  {element:20s}: {count:4d} present  |  {missing:3d} missing")


    combo_counter = Counter()
    for d in dataset:
        present = tuple(sorted(e for e in PICO_ELEMENTS if e in d['labels']))
        combo_counter[len(present)] += 1
    print(f"\n  Abstracts with all 3 elements : {combo_counter[3]}")
    print(f"  Abstracts with 2 elements     : {combo_counter[2]}")
    print(f"  Abstracts with 1 element      : {combo_counter[1]}")
    print(f"  Abstracts with 0 elements     : {combo_counter[0]}")


    print(f"  4. TOKEN–LABEL ALIGNMENT CHECK")

    mismatches = 0
    for d in dataset:
        for element in PICO_ELEMENTS:
            if element in d['labels']:
                if len(d['labels'][element]) != len(d['tokens']):
                    mismatches += 1
    print(f"  Total mismatches found : {mismatches}")
    print(f"  (These were already excluded during loading)")


    
    print(f"  5. SAMPLE ABSTRACT")
    
    sample = next((d for d in dataset if len(d['labels']) == 3), dataset[0])
    print(f"\n  PMID  : {sample['pmid']}")
    print(f"  Split : {sample['split']}")
    print(f"  Tokens: {len(sample['tokens'])}")
    print(f"\n  Text:\n  {sample['text'][:400]}...\n")

    for element in PICO_ELEMENTS:
        if element in sample['labels']:
            spans = extract_spans(sample['tokens'], sample['labels'][element])
            print(f"  {element.upper()} spans:")
            for s in spans[:3]:
                print(f"    → {s}")
        else:
            print(f"  {element.upper()} spans: (none)")
        print()


#MAIN
if __name__ == '__main__':
    print("Loading dataset")
    dataset = load_all_documents()

    if len(dataset) == 0:
        print("ERROR: No documents loaded.")
        print(f"  Current DATA_DIR = {DATA_DIR}")
    else:
        explore(dataset)
        print("\n Dataset exploration complete.")