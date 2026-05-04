# Structured Information Extraction from Clinical Trial Abstracts

EMATM0067 — Group 32

Extracting structured PICO (Participants, Interventions, Outcomes) information from clinical trial abstracts using the EBM-NLP dataset. The project compares rule-based, LLM-based, and supervised extraction pipelines across two design axes.



## Repository Structure


├── cleaned_data/
│   ├── cleaned_dataset.json       
│   ├── cleaned_summary.csv       
│   ├── train.json                 # Training split (4,742 abstracts)
│   └── test.json                  # Test split (191 abstracts)
│
├── ebm_nlp_2_00/
│   ├── annotations/               # Raw EBM-NLP token-level annotations
│   └── documents/                 # Raw abstract text files
│
├── results/
│   ├── axis1/
│   │   ├── extraction_pipeline_results.json    # Rule-based and LLM results
│   │   └── semantic_extended_results.json      # Semantic evaluation scores
│   └── axis2/
│       ├── BiomedBERT_two_stage_decomposed/    # Decomposed pipeline results
│       ├── Token-Classification-BiomedBERT/    # End-to-end pipeline results
│       ├── semantic_decompose-BiomedBERT/      # Decomposed semantic scores
│       └── semantic_extended-BiomedBERT/       # End-to-end semantic scores
│
├── step1_data_preparation/
│   ├── clean_data.py              
│   ├── explore_data.py           
│   └── save_cleaned_data.py       # Exports cleaned data to JSON/CSV
│
├── step2_clustering/
│   └── Clustering.ipynb           # K-means and HAC sentence clustering
│
├── step3_extraction/
│   ├── axis1/                     # Rule-based and LLM extraction scripts
│   └── axis2/
│       ├── end-to-end_biomedbert.py            # End-to-end BiomedBERT pipeline
│       └── decomposed_biomedbert_gpu.py        # Decomposed BiomedBERT pipeline
│
├──step4_evaluation/
|    ├── semantic_evaluation.py     # Cosine similarity evaluation framework
└──ebm_nlp_2_00.tar.gz              



## Pipeline Overview

The project runs in four sequential steps:

Step 1 — Data Preparation:
Load raw EBM-NLP files, validate token-label alignment, filter short/unlabelled abstracts, extract readable PICO spans, and export a unified JSON dataset.

Step 2 — Clustering:
Apply K-means and HAC to sentence embeddings to test whether sentences naturally separate into P/I/O groups. Results show poor separability 

Step 3 — Extraction:
- Axis 1: Rule-based keyword matching vs. LLM extraction (Claude Haiku 4.5) with 0-shot/5-shot and end-to-end/decomposed prompting — 5 pipelines total  
- Axis 2: End-to-end BiomedBERT token classification vs. decomposed sentence router + token extractor

Step 4 — Evaluation:
Semantic similarity matching using `all-MiniLM-L6-v2` sentence transformer with a cosine similarity threshold of 0.75. Micro-averaged precision, recall, and F1 reported per field.


## Requirements:

pip install anthropic sentence-transformers transformers torch nltk scikit-learn

