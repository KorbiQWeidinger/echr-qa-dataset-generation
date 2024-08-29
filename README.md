# ECHR-QA Dataset Generation

This repository contains all code used to curate the ECHR-QA dataset.

The dataset creation can be reproduced using the following steps:

1. Parse the case law guides at sentence level such that a db with guide_id, paragraph, sentence, citations is obtained `data/sentences_with_citations.csv`
2. Curate QA pairs using `question_generation.py`

Note: 1. included a lot of individual steps including some manual citation mappings as citation extraction is a complex task.
If I had to do this again I would probably try to use a better PDF parser and include LLMs for identifying indirect citations.

## Expert Annotations

Expert annotations can be found in `data/annotated_qa_pairs.csv`.

## ECHR-QA Dataset

The final dataset can be found under `data/echr_qa_dataset.csv`.

## Development Instructions

```
python -m venv env
source env/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=value
deactivate
```
