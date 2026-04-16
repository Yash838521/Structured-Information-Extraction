import os
import re
import sys
import json
import time
import numpy as np
from collections import defaultdict


ELEMENTS = ['participants', 'interventions', 'outcomes']


#  DATA LOADING

def load_splits(data_dir='cleaned_data'):
    with open(os.path.join(data_dir, 'train.json')) as f:
        train = json.load(f)
    with open(os.path.join(data_dir, 'test.json')) as f:
        test = json.load(f)
    return train, test


def split_sentences(text):
    raw = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in raw if len(s.strip()) > 3]


#  PIPELINE 1 — RULE-BASED EXTRACTION
PARTICIPANT_CUES = {
    'patients', 'subjects', 'participants', 'children', 'adults', 'women',
    'men', 'volunteers', 'individuals', 'infants', 'adolescents', 'elderly',
    'enrolled', 'recruited', 'randomized', 'randomised', 'aged', 'diagnosed',
    'healthy', 'male', 'female', 'obese', 'pregnant', 'undergoing',
    'consecutive', 'outpatients', 'inpatients', 'population', 'cohort',
    'screened', 'eligible', 'included', 'excluded', 'males', 'females',
    'newborns', 'neonates', 'preschool', 'postmenopausal'
}

INTERVENTION_CUES = {
    'treatment', 'therapy', 'drug', 'dose', 'placebo', 'intervention',
    'administered', 'received', 'mg', 'daily', 'twice', 'oral', 'topical',
    'intravenous', 'injection', 'surgery', 'versus', 'compared', 'plus',
    'combination', 'regimen', 'assigned', 'allocated', 'saline', 'control',
    'capsule', 'tablet', 'infusion', 'subcutaneous', 'extract', 'supplement',
    'vaccine', 'antibiotic', 'steroid', 'inhibitor', 'agonist', 'antagonist',
    'blocker', 'counseling', 'counselling', 'program', 'programme',
    'exercise', 'training', 'acupuncture', 'massage', 'physiotherapy'
}

OUTCOME_CUES = {
    'outcome', 'efficacy', 'safety', 'response', 'survival', 'mortality',
    'improvement', 'reduction', 'score', 'rate', 'adverse', 'events',
    'pain', 'quality', 'function', 'levels', 'change', 'difference',
    'endpoint', 'measure', 'tolerability', 'remission', 'relapse', 'death',
    'recurrence', 'complication', 'satisfaction', 'duration', 'incidence',
    'prevalence', 'symptom', 'symptoms', 'cure', 'healing', 'recovery',
    'biomarker', 'concentration', 'clearance', 'eradication', 'morbidity',
    'hospitalization', 'readmission', 'infection'
}

SECTION_HEADERS = {
    'background': re.compile(r'\b(BACKGROUND|INTRODUCTION|CONTEXT|OBJECTIVE|AIM|PURPOSE|AIMS)\b'),
    'methods':    re.compile(r'\b(METHOD|METHODS|DESIGN|SETTING|MATERIAL|MATERIALS|PROCEDURE|PROTOCOL|PATIENTS)\b'),
    'results':    re.compile(r'\b(RESULT|RESULTS|FINDING|FINDINGS)\b'),
    'conclusion': re.compile(r'\b(CONCLUSION|CONCLUSIONS|IMPLICATION|SUMMARY)\b'),
}


class RuleBasedPipeline:

    def __init__(self):
        self.name = "Rule-based"

    def train(self, train_data):
        pass

    def extract(self, doc):
        text = doc['text']
        sentences = split_sentences(text)
        n = len(sentences)
        if n == 0:
            return {e: [] for e in ELEMENTS}

        predictions = {e: [] for e in ELEMENTS}

        for i, sent in enumerate(sentences):
            words = set(re.findall(r'[a-zA-Z]+', sent.lower()))
            rel_pos = i / max(n - 1, 1)

        
            in_methods = bool(SECTION_HEADERS['methods'].search(sent))
            in_results = bool(SECTION_HEADERS['results'].search(sent))

            
            p_hits = len(words & PARTICIPANT_CUES)
            
            if p_hits >= 2:
                predictions['participants'].append(sent)
            elif p_hits >= 1 and (rel_pos < 0.5 or in_methods):
                predictions['participants'].append(sent)

            
            i_hits = len(words & INTERVENTION_CUES)
            if i_hits >= 2:
                predictions['interventions'].append(sent)
            elif i_hits >= 1 and in_methods:
                predictions['interventions'].append(sent)

            
            o_hits = len(words & OUTCOME_CUES)
            if o_hits >= 2:
                predictions['outcomes'].append(sent)
            elif o_hits >= 1 and (rel_pos > 0.4 or in_results):
                predictions['outcomes'].append(sent)

        return predictions



#  PIPELINE 2 — LLM END-TO-END (single prompt, all fields at once)

END_TO_END_PROMPT = """You are a biomedical research assistant. Your job is to read a clinical trial abstract and extract three pieces of structured information from it.

DEFINITIONS:
- Participants: Who was studied? Look for descriptions of the study population — their condition, demographics, sample size, eligibility criteria. Examples: "patients with type 2 diabetes", "healthy adult volunteers aged 18-65", "children with asthma".
- Interventions: What treatments or procedures were compared? Look for drugs, therapies, dosages, surgical techniques, or behavioural programs. Examples: "metformin 500mg twice daily", "cognitive behavioural therapy", "laparoscopic surgery versus open surgery".
- Outcomes: What was measured or assessed? Look for clinical endpoints, test scores, biomarkers, adverse events. Examples: "overall survival", "pain score on VAS", "rate of adverse events", "HbA1c levels".


Abstract: "{abstract}"

Return ONLY a JSON object with keys "participants", "interventions", "outcomes", each containing a list of short extracted phrases. Extract the most specific phrases you can find — do not copy entire sentences. If a field is not mentioned, use an empty list."""


class LLMEndToEndPipeline:

    def __init__(self, api_key=None):
        self.name = "LLM-EndToEnd"
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY', '')

    def train(self, train_data):
        pass

    def _call_api(self, prompt):
        """Call the Claude API and return the text response."""
        import urllib.request

        body = json.dumps({
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 1000,
            "messages": [{"role": "user", "content": prompt}]
        })

        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=body.encode('utf-8'),
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01"
            }
        )

        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode('utf-8'))

       
        text_parts = [block['text'] for block in data['content'] if block['type'] == 'text']
        return ' '.join(text_parts)

    def _parse_json_response(self, response_text):
        """Extract the JSON object from the LLM's response."""
       
        try:
            return json.loads(response_text.strip())
        except json.JSONDecodeError:
            pass

       
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        
        match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        return None

    def extract(self, doc):
        prompt = END_TO_END_PROMPT.replace("{abstract}", doc['text'])

        try:
            response = self._call_api(prompt)
            parsed = self._parse_json_response(response)

            if parsed:
                result = {}
                for elem in ELEMENTS:
                    val = parsed.get(elem, [])
                    if isinstance(val, str):
                        val = [val]
                    result[elem] = val
                return result

        except Exception as e:
            print(f"    API error for PMID {doc['pmid']}: {e}")

        return {e: [] for e in ELEMENTS}



#  PIPELINE 3 — LLM DECOMPOSED (three separate prompts, one per field)

DECOMPOSED_PROMPTS = {
    'participants': """You are extracting PARTICIPANT information from a clinical trial abstract.

Participants = who was studied. Look for:
- The medical condition or disease being studied
- Demographics (age, sex, specific populations)
- Sample size (number of patients/subjects)
- Eligibility criteria

Example:
Abstract: "Sixty adult patients, ASA Classes I & II, were involved in a study to compare endotracheal tubes during intranasal surgery."
Output: ["Sixty adult patients, ASA Classes I & II"]

Now extract from this abstract. Return ONLY a JSON list of short phrases.

Abstract: "{abstract}"
""",

    'interventions': """You are extracting INTERVENTION information from a clinical trial abstract.

Interventions = what treatments, drugs, or procedures were tested. Look for:
- Drug names and dosages
- Surgical or therapeutic procedures
- Comparators (placebo, standard care, alternative treatment)
- Treatment duration or regimen

Example:
Abstract: "Twenty-nine patients were randomly allocated to oral treatment with either ketoconazole 200 mg daily or griseofulvin 1 g daily for up to 8 weeks."
Output: ["ketoconazole 200 mg daily", "griseofulvin 1 g daily"]

Now extract from this abstract. Return ONLY a JSON list of short phrases.

Abstract: "{abstract}"
""",

    'outcomes': """You are extracting OUTCOME information from a clinical trial abstract.

Outcomes = what was measured or assessed. Look for:
- Primary and secondary endpoints
- Clinical measurements (blood pressure, survival, cure rate)
- Patient-reported measures (pain scores, quality of life)
- Safety outcomes (adverse events, side effects)

Example:
Abstract: "Vaccine effectiveness were tested by counting active PU lesions, serum eosinophils, and IgE before and after 4 months of treatment."
Output: ["active PU lesions", "serum eosinophils", "IgE"]

Now extract from this abstract. Return ONLY a JSON list of short phrases.

Abstract: "{abstract}"
"""
}


class LLMDecomposedPipeline:

    def __init__(self, api_key=None):
        self.name = "LLM-Decomposed"
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY', '')

    def train(self, train_data):
        pass

    def _call_api(self, prompt):
        import urllib.request

        body = json.dumps({
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 500,
            "messages": [{"role": "user", "content": prompt}]
        })

        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=body.encode('utf-8'),
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01"
            }
        )

        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode('utf-8'))

        text_parts = [block['text'] for block in data['content'] if block['type'] == 'text']
        return ' '.join(text_parts)

    def _parse_list_response(self, response_text):
        """Parse a JSON list from the response."""
        # Direct parse
        try:
            result = json.loads(response_text.strip())
            if isinstance(result, list):
                return result
            if isinstance(result, dict):
                # Sometimes the model wraps it in a dict
                for v in result.values():
                    if isinstance(v, list):
                        return v
        except json.JSONDecodeError:
            pass

        match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', response_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Look for any JSON list
        match = re.search(r'\[.*?\]', response_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        return []

    def extract(self, doc):
        results = {}

        for elem in ELEMENTS:
            prompt = DECOMPOSED_PROMPTS[elem].replace("{abstract}", doc['text'])

            try:
                response = self._call_api(prompt)
                spans = self._parse_list_response(response)
                results[elem] = [s for s in spans if isinstance(s, str)]
            except Exception as e:
                print(f"    API error ({elem}) for PMID {doc['pmid']}: {e}")
                results[elem] = []

            # Small delay to avoid rate limiting
            time.sleep(0.3)

        return results




def compute_token_overlap(predicted_spans, gold_spans):
    
    pred_words = set()
    for span in predicted_spans:
        for w in span.lower().split():
            if w.isalpha():
                pred_words.add(w)

    gold_words = set()
    for span in gold_spans:
        for w in span.lower().split():
            if w.isalpha():
                gold_words.add(w)

    if not pred_words and not gold_words:
        return 1.0, 1.0, 1.0
    if not pred_words:
        return 0.0, 0.0, 0.0
    if not gold_words:
        return 0.0, 0.0, 0.0

    overlap   = pred_words & gold_words
    precision = len(overlap) / len(pred_words)
    recall    = len(overlap) / len(gold_words)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def evaluate_pipeline(pipeline, test_data, save_predictions=False):
    
    results = {e: {'precisions': [], 'recalls': [], 'f1s': [], 'coverage': 0}
               for e in ELEMENTS}
    all_predictions = []

    for idx, doc in enumerate(test_data):
        if idx % 20 == 0:
            print(f"      processing {idx+1}/{len(test_data)}...")

        preds = pipeline.extract(doc)

        if save_predictions:
            all_predictions.append({
                'pmid': doc['pmid'],
                'predictions': preds,
                'gold': doc['spans']
            })

        for elem in ELEMENTS:
            gold = doc['spans'][elem]
            if not gold:
                continue

            p, r, f = compute_token_overlap(preds[elem], gold)
            results[elem]['precisions'].append(p)
            results[elem]['recalls'].append(r)
            results[elem]['f1s'].append(f)

            if len(preds[elem]) > 0:
                results[elem]['coverage'] += 1

    
    summary = {}
    for elem in ELEMENTS:
        n = len(results[elem]['f1s'])
        if n == 0:
            summary[elem] = {'precision': 0, 'recall': 0, 'f1': 0, 'coverage': 0, 'n': 0}
        else:
            summary[elem] = {
                'precision': round(float(np.mean(results[elem]['precisions'])), 4),
                'recall':    round(float(np.mean(results[elem]['recalls'])), 4),
                'f1':        round(float(np.mean(results[elem]['f1s'])), 4),
                'coverage':  round(results[elem]['coverage'] / n, 4),
                'n':         n
            }

    return summary, all_predictions


def print_results_table(all_results):
    print(f"\n{'='*88}")
    print(f"  EVALUATION RESULTS")
    print(f"{'='*88}")
    print(f"  {'Pipeline':<22s} {'Element':<16s} {'Prec':>7s} {'Rec':>7s} {'F1':>7s} {'Cov%':>7s}  {'N':>4s}")
    print(f"  {'-'*82}")

    for pname, summary in all_results.items():
        for i, elem in enumerate(ELEMENTS):
            s = summary[elem]
            label = pname if i == 0 else ''
            print(f"  {label:<22s} {elem:<16s} {s['precision']:>7.3f} {s['recall']:>7.3f} "
                  f"{s['f1']:>7.3f} {s['coverage']*100:>6.1f}%  {s['n']:>4d}")
        print(f"  {'-'*82}")

    # Macro averages
    print(f"\n  {'Pipeline':<22s} {'Macro-F1':>9s} {'Macro-Prec':>11s} {'Macro-Rec':>10s}")
    print(f"  {'-'*56}")
    for pname, summary in all_results.items():
        mf = np.mean([summary[e]['f1'] for e in ELEMENTS])
        mp = np.mean([summary[e]['precision'] for e in ELEMENTS])
        mr = np.mean([summary[e]['recall'] for e in ELEMENTS])
        print(f"  {pname:<22s} {mf:>9.3f} {mp:>11.3f} {mr:>10.3f}")



#  DOWNSTREAM USABILITY

def run_downstream_queries(pipeline_name, predictions_list, test_data):
   
    print(f"\n  Downstream usability — {pipeline_name}")
    print(f"  {'-'*55}")

    pred_lookup = {p['pmid']: p['predictions'] for p in predictions_list}

    queries = [
        ("Trials involving children",
         'participants', {'children', 'child', 'pediatric', 'paediatric', 'infant', 'adolescent', 'neonatal'}),
        ("Trials with placebo",
         'interventions', {'placebo'}),
        ("Trials measuring survival/mortality",
         'outcomes', {'survival', 'mortality', 'death', 'died'}),
    ]

    for qname, elem, keywords in queries:
        gold_hits = set()
        pred_hits = set()

        for doc in test_data:
            pmid = doc['pmid']
            gold_text = ' '.join(doc['spans'][elem]).lower()
            if any(k in gold_text for k in keywords):
                gold_hits.add(pmid)

            if pmid in pred_lookup:
                pred_text = ' '.join(pred_lookup[pmid].get(elem, [])).lower()
                if any(k in pred_text for k in keywords):
                    pred_hits.add(pmid)

        if gold_hits:
            tp = len(pred_hits & gold_hits)
            fp = len(pred_hits - gold_hits)
            fn = len(gold_hits - pred_hits)
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            print(f"    {qname:<38s} gold={len(gold_hits):3d}  "
                  f"found={len(pred_hits):3d}  prec={p:.2f}  rec={r:.2f}")
        else:
            print(f"    {qname:<38s} (no gold matches in test set)")

#  MAIN

def main():
    print("Loading data...")
    train, test = load_splits()
    print(f"  Train: {len(train)},  Test: {len(test)}\n")

    api_key = os.environ.get('ANTHROPIC_API_KEY', '')
    has_api = len(api_key) > 10

    if not has_api:
        print("  NOTE: No ANTHROPIC_API_KEY found in environment.")
        print("  → Rule-based pipeline will run fully.")
        print("  → LLM pipelines will be skipped.")
        print("  → Set the key with: export ANTHROPIC_API_KEY='your-key-here'")
        print()

    # Pipeline 1: Rule-based
    print("  PIPELINE 1: RULE-BASED")
    rule_pipe = RuleBasedPipeline()
    rule_pipe.train(train)

    print("  Evaluating...")
    rule_summary, rule_preds = evaluate_pipeline(rule_pipe, test, save_predictions=True)
    print("  Done.\n")

    all_results = {'Rule-based': rule_summary}
    all_preds   = {'Rule-based': rule_preds}

     
    if has_api:
        print("Connecting to the API")
        try:
            import urllib.request
            test_body = json.dumps({
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 50,
                "messages": [{"role": "user", "content": "Say OK"}]
            })
            test_req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=test_body.encode('utf-8'),
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01"
                }
            )
            with urllib.request.urlopen(test_req, timeout=15) as resp:
                test_data = json.loads(resp.read().decode('utf-8'))
            print("  API connection successful!\n")
        except Exception as e:
            print(f"  API test FAILED: {e}")
            print("  Skipping LLM pipelines. Check your API key and internet connection.")
            has_api = False
            print()

    #  Pipeline 2: LLM End-to-End 
    if has_api:
        print("  PIPELINE 2: LLM END-TO-END")
        e2e_pipe = LLMEndToEndPipeline(api_key=api_key)
        e2e_pipe.train(train)

        print("  Evaluating (this will take a few minutes)...")
        e2e_summary, e2e_preds = evaluate_pipeline(e2e_pipe, test, save_predictions=True)
        print("  Done.\n")

        all_results['LLM-EndToEnd'] = e2e_summary
        all_preds['LLM-EndToEnd']   = e2e_preds

    # Pipeline 3: LLM Decomposed 
    if has_api:
        print("  PIPELINE 3: LLM DECOMPOSED")
        dec_pipe = LLMDecomposedPipeline(api_key=api_key)
        dec_pipe.train(train)

        print("Evaluating")
        dec_summary, dec_preds = evaluate_pipeline(dec_pipe, test, save_predictions=True)
        print("  Finished\n")

        all_results['LLM-Decomposed'] = dec_summary
        all_preds['LLM-Decomposed']   = dec_preds

    
    print_results_table(all_results) 

    
    print(f"\n{'='*88}")
    print(f"  DOWNSTREAM USABILITY TESTS")
    print(f"{'='*88}")
    for pname, preds in all_preds.items():
        run_downstream_queries(pname, preds, test)

    

    #  Saving all results to JSON 
    output = {
        'summaries': all_results,
        'predictions': {
            name: [
                {'pmid': p['pmid'], 'predictions': p['predictions'], 'gold': p['gold']}
                for p in preds
            ]
            for name, preds in all_preds.items()
        }
    }

    output_path = 'extraction_pipeline_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  All results saved to: {output_path}")


if __name__ == '__main__':
    main()