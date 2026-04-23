
import os
import json
import time
import re
import sys
from nltk.tokenize import sent_tokenize


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_prompts import (
    END_TO_END_PROMPT_0_SHOT,
    END_TO_END_PROMPT_MULTISHOT,
    DECOMPOSED_PROMPTS_0_SHOT,
    DECOMPOSED_PROMPTS_MULTISHOT,
)



PROJECT_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DATA_PATH = os.path.join(PROJECT_ROOT, "cleaned_data", "test.json")
OUTPUT_PATH    = os.path.join(PROJECT_ROOT, "results", "extraction_pipeline_results.json")
MODEL          = "claude-haiku-4-5-20251001"
FIELDS         = ["participants", "interventions", "outcomes"]



def load_test_data():
    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)



P_CUES = {
    'patients', 'subjects', 'participants', 'children', 'adults', 'women',
    'men', 'volunteers', 'individuals', 'infants', 'adolescents', 'elderly',
    'enrolled', 'recruited', 'randomized', 'randomised', 'aged', 'diagnosed',
    'healthy', 'male', 'female', 'obese', 'pregnant', 'undergoing',
    'consecutive', 'outpatients', 'inpatients', 'population', 'cohort',
    'screened', 'eligible', 'included', 'excluded', 'males', 'females',
    'newborns', 'neonates', 'preschool', 'postmenopausal'
}

I_CUES = {
    'treatment', 'therapy', 'drug', 'dose', 'placebo', 'intervention',
    'administered', 'received', 'mg', 'daily', 'twice', 'oral', 'topical',
    'intravenous', 'injection', 'surgery', 'versus', 'compared', 'plus',
    'combination', 'regimen', 'assigned', 'allocated', 'saline', 'control',
    'capsule', 'tablet', 'infusion', 'subcutaneous', 'extract', 'supplement',
    'vaccine', 'antibiotic', 'steroid', 'inhibitor', 'agonist', 'antagonist',
    'blocker', 'counseling', 'counselling', 'program', 'programme',
    'exercise', 'training', 'acupuncture', 'massage', 'physiotherapy'
}

O_CUES = {
    'outcome', 'efficacy', 'safety', 'response', 'survival', 'mortality',
    'improvement', 'reduction', 'score', 'rate', 'adverse', 'events',
    'pain', 'quality', 'function', 'levels', 'change', 'difference',
    'endpoint', 'measure', 'tolerability', 'remission', 'relapse', 'death',
    'recurrence', 'complication', 'satisfaction', 'duration', 'incidence',
    'prevalence', 'symptom', 'symptoms', 'cure', 'healing', 'recovery',
    'biomarker', 'concentration', 'clearance', 'eradication', 'morbidity',
    'hospitalization', 'readmission', 'infection'
}

CUE_MAP = {
    "participants":  P_CUES,
    "interventions": I_CUES,
    "outcomes":      O_CUES,
}


def rule_based_extract(text):
    sentences = sent_tokenize(text)
    predictions = {f: [] for f in FIELDS}
    for sentence in sentences:
        words = set(sentence.lower().split())
        for field in FIELDS:
            if words & CUE_MAP[field]:
                predictions[field].append(sentence)
    return predictions


# LLM EXTRACTOR 

def create_client():
    try:
        import anthropic
    except ImportError:
        print("Installing anthropic package...")
        os.system(f"{sys.executable} -m pip install anthropic")
        import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY not set.\n"
            "  PowerShell:  $env:ANTHROPIC_API_KEY=\"sk-ant-...\"\n"
            "  CMD:         set ANTHROPIC_API_KEY=sk-ant-..."
        )
    return anthropic.Anthropic(api_key=api_key)


def call_llm(client, prompt, max_retries=5):
    import anthropic

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text

        except anthropic.RateLimitError:
            wait = min(30 * (attempt + 1), 120)
            time.sleep(wait)

        except anthropic.APIStatusError as e:
            if e.status_code == 529:
                wait = 30 * (attempt + 1)
                time.sleep(wait)
            else:
                print(f"    API error ({e.status_code}): {e.message}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                else:
                    return None

        except Exception as e:
            print(f"    Unexpected error: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                return None

    return None


def parse_json_response(response_text):
    if response_text is None:
        return None
    cleaned = re.sub(r"json\s*", "", response_text)
    cleaned = re.sub(r"\s*", "", cleaned)
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r'[\[{].*[\]}]', cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        print(f"    Warning: Could not parse response")
        return None


def llm_end_to_end_extract(client, abstract, prompt_template):
    prompt = prompt_template.replace("{abstract}", abstract)
    response = call_llm(client, prompt)
    parsed = parse_json_response(response)
    if parsed and isinstance(parsed, dict):
        return {
            "participants":  parsed.get("participants", []),
            "interventions": parsed.get("interventions", []),
            "outcomes":      parsed.get("outcomes", []),
        }
    return {"participants": [], "interventions": [], "outcomes": []}


def llm_decomposed_extract(client, abstract, prompt_dict):
    predictions = {}
    for field in FIELDS:
        prompt = prompt_dict[field].replace("{abstract}", abstract)
        response = call_llm(client, prompt)
        parsed = parse_json_response(response)
        if parsed and isinstance(parsed, list):
            predictions[field] = parsed
        elif parsed and isinstance(parsed, dict):
            predictions[field] = parsed.get(field, [])
        else:
            predictions[field] = []
        time.sleep(0.5)
    return predictions


# CHECKPOINT

CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "results", "_checkpoint.json")


def save_checkpoint(all_predictions, current_pipeline, current_index):
    checkpoint = {
        "predictions": all_predictions,
        "current_pipeline": current_pipeline,
        "current_index": current_index,
    }
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(checkpoint, f)


def load_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "r") as f:
            return json.load(f)
    return None


# PIPELINE EXECUTION

def run_all_pipelines(test_data):
    llm_pipelines = {
        "LLM-EndToEnd-0shot":    ("end_to_end", END_TO_END_PROMPT_0_SHOT),
        "LLM-EndToEnd-5shot":    ("end_to_end", END_TO_END_PROMPT_MULTISHOT),
        "LLM-Decomposed-0shot":  ("decomposed", DECOMPOSED_PROMPTS_0_SHOT),
        "LLM-Decomposed-5shot":  ("decomposed", DECOMPOSED_PROMPTS_MULTISHOT),
    }

    
    checkpoint = load_checkpoint()
    if checkpoint:
        all_predictions = checkpoint["predictions"]
        skip_to_pipeline = checkpoint["current_pipeline"]
        skip_to_index = checkpoint["current_index"]
        print(f"  Resuming from checkpoint: {skip_to_pipeline}, doc {skip_to_index}")
    else:
        all_predictions = {}
        skip_to_pipeline = None
        skip_to_index = 0

    #  Pipeline 1: Rule-based 
    if "Rule-based" not in all_predictions:
       
        print(f"  Running: Rule-based")

        rule_based_results = []
        for i, doc in enumerate(test_data):
            preds = rule_based_extract(doc["text"])
            rule_based_results.append({
                "pmid":        doc["pmid"],
                "predictions": preds,
                "gold":        doc["spans"],
            })
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(test_data)} documents")

        all_predictions["Rule-based"] = rule_based_results
        print(f"  Done: {len(rule_based_results)} documents")
    else:
        print(f"\n  Skipping Rule-based already completed")

    # Pipelines 2-5: LLM variants
    client = create_client()
    found_resume_point = (skip_to_pipeline is None)

    for pipeline_name, (mode, prompt_config) in llm_pipelines.items():

        if not found_resume_point:
            if pipeline_name == skip_to_pipeline:
                found_resume_point = True
            elif pipeline_name in all_predictions:
                print(f"\n  Skipping {pipeline_name} already completed")
                continue

    
        print(f"  Running: {pipeline_name}")
        

        if pipeline_name in all_predictions and skip_to_pipeline == pipeline_name:
            pipeline_results = all_predictions[pipeline_name]
            start_index = skip_to_index
            print(f"  Resuming from document {start_index}")
        else:
            pipeline_results = []
            start_index = 0

        for i in range(start_index, len(test_data)):
            doc = test_data[i]
            abstract = doc["text"]

            if mode == "end_to_end":
                preds = llm_end_to_end_extract(client, abstract, prompt_config)
            else:
                preds = llm_decomposed_extract(client, abstract, prompt_config)

            pipeline_results.append({
                "pmid":        doc["pmid"],
                "predictions": preds,
                "gold":        doc["spans"],
            })

            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(test_data)} documents")
                all_predictions[pipeline_name] = pipeline_results
                save_checkpoint(all_predictions, pipeline_name, i + 1)

            time.sleep(1)

        all_predictions[pipeline_name] = pipeline_results
        print(f"  Done: {len(pipeline_results)} documents")

    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)

    return all_predictions


# COMPUTE METRICS

def compute_token_overlap(pred_spans, gold_spans):
    if not pred_spans or not gold_spans:
        return 0.0, 0.0, 0.0
    pred_tokens = set()
    for span in pred_spans:
        pred_tokens.update(span.lower().split())
    gold_tokens = set()
    for span in gold_spans:
        gold_tokens.update(span.lower().split())
    overlap = pred_tokens & gold_tokens
    precision = len(overlap) / len(pred_tokens) if pred_tokens else 0
    recall    = len(overlap) / len(gold_tokens) if gold_tokens else 0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0)
    return precision, recall, f1


def compute_summaries(all_predictions):
    summaries = {}
    for pipeline_name, docs in all_predictions.items():
        field_metrics = {}
        for field in FIELDS:
            precisions, recalls, f1s = [], [], []
            n_with_gold = 0
            for doc in docs:
                gold = doc["gold"].get(field, [])
                if not gold:
                    continue
                n_with_gold += 1
                pred = doc["predictions"].get(field, [])
                p, r, f = compute_token_overlap(pred, gold)
                precisions.append(p)
                recalls.append(r)
                f1s.append(f)
            if n_with_gold > 0:
                coverage = sum(
                    1 for doc in docs
                    if doc["predictions"].get(field, [])
                ) / len(docs)
                field_metrics[field] = {
                    "precision": round(sum(precisions) / len(precisions), 4),
                    "recall":    round(sum(recalls) / len(recalls), 4),
                    "f1":        round(sum(f1s) / len(f1s), 4),
                    "coverage":  round(coverage, 4),
                    "n":         n_with_gold,
                }
        summaries[pipeline_name] = field_metrics
    return summaries


# MAIN EXECUTION
def main():
    print("Load test data")
    test_data = load_test_data()
    print(f"Loaded {len(test_data)} test documents")
    print(f"Model: {MODEL}")
   
    all_predictions = run_all_pipelines(test_data)
    summaries = compute_summaries(all_predictions)

    output = {
        "summaries":   summaries,
        "predictions": all_predictions,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"  RESULTS SAVED TO: {OUTPUT_PATH}")

    for name, summary in summaries.items():
        print(f"\n  {name}:")
        for field, metrics in summary.items():
            print(f"    {field}: P={metrics['precision']:.3f}  "
                  f"R={metrics['recall']:.3f}  F1={metrics['f1']:.3f}")


if __name__ == "__main__":
    main()