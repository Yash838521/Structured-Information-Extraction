import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")


def load_results(path=None):
    if path is None:
        path = os.path.join(RESULTS_DIR, "extraction_pipeline_results.json")
    with open(path, "r") as f:
        return json.load(f)


class SemanticEvaluator:
    def __init__(self):
        print("Load embedding model")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def score(self, pred_spans, gold_spans):

        if not pred_spans and not gold_spans:
            return 1.0, 1.0, 1.0, 0, 0, 0
        if not pred_spans:
            return 0.0, 0.0, 0.0, 0, len(gold_spans), 0
        if not gold_spans:
            return 0.0, 0.0, 0.0, len(pred_spans), 0, 0

        pred_emb = self.model.encode(pred_spans) #text to semantic embeddings
        gold_emb = self.model.encode(gold_spans) # text to semantic embeddings
 
        sim = cosine_similarity(pred_emb, gold_emb)

        matched_pred = set()
        matched_gold = set()

        # Matching the predictions to gold predictions if the similarity score is above a certain threshold:
        for i in range(sim.shape[0]):
            j = sim[i].argmax()
            if sim[i][j] > 0.75:
                matched_pred.add(i)
                matched_gold.add(j)

        
        tp = len(matched_pred)
        fp = len(pred_spans) - tp
        fn = len(gold_spans) - len(matched_gold)

        precision = tp / len(pred_spans) if pred_spans else 0.0
        recall = tp / len(gold_spans) if gold_spans else 0.0
        f1 = (2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0)

        return precision, recall, f1, tp, fp, fn

    def evaluate_pipeline(self, preds):

        # Create 3 separate fields 
        fields = ["participants", "interventions", "outcomes"]

        results = {f: [] for f in fields}

        global_tp = 0
        global_fp = 0
        global_fn = 0

        per_doc_f1s = []

        total_pred = {f: 0 for f in fields}
        total_gold = {f: 0 for f in fields}

        field_tp = {f: 0 for f in fields}
        field_fp = {f: 0 for f in fields}
        field_fn = {f: 0 for f in fields}

        
        for doc in preds:

            pred = doc["predictions"]
            gold = doc["gold"]

            doc_f1s = []

            for field in fields:
                p, r, f, tp, fp, fn = self.score(
                    pred.get(field, []),
                    gold.get(field, [])
                )
                results[field].append((p, r, f))

                global_tp += tp
                global_fp += fp
                global_fn += fn

                field_tp[field] += tp
                field_fp[field] += fp
                field_fn[field] += fn

                total_pred[field] += len(pred.get(field, []))
                total_gold[field] += len(gold.get(field, []))
                doc_f1s.append(f)

            per_doc_f1s.append(np.mean(doc_f1s))

        summary = {}

        # Converting the stored scores into an array and compute the average and stats
        for field, scores in results.items():

            arr = np.array(scores)

            summary[field] = {
                "precision": float(np.mean(arr[:, 0])),
                "recall": float(np.mean(arr[:, 1])),
                "f1": float(np.mean(arr[:, 2])),

                "false_positives": int(field_fp[field]),
                "false_negatives": int(field_fn[field]),

                "prediction_volume": int(total_pred[field]),
                "gold_volume": int(total_gold[field]),

                "std_f1": float(np.std(arr[:, 2])),
                "n_docs": len(scores)
            }

        micro_precision = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0
        micro_recall = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0

        micro_f1 = (
            2 * micro_precision * micro_recall /
            (micro_precision + micro_recall)
            if (micro_precision + micro_recall) > 0 else 0
        )

       
        summary["_GLOBAL"] = {
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_f1": micro_f1,
            "total_tp": global_tp,
            "total_fp": global_fp,
            "total_fn": global_fn,
        }

        
        summary["_CONSISTENCY"] = {
            "mean_f1": float(np.mean(per_doc_f1s)),
            "std_f1_per_doc": float(np.std(per_doc_f1s))
        }

        return summary

# Save the results
def save_results(all_results, output_path=None):
    
    if output_path is None:
        output_path = os.path.join(RESULTS_DIR, "semantic_extended_results.json")

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSaved results to: {output_path}")


def main():

    data = load_results()
    evaluator = SemanticEvaluator()

    all_results = {}

    for pipeline_name, preds in data["predictions"].items():

        print(f"Evaluating: {pipeline_name}...")
        results = evaluator.evaluate_pipeline(preds)
        all_results[pipeline_name] = results

    save_results(all_results)


if __name__ == "__main__":
    main()