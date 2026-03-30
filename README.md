# HIFI-KPI: A Dataset for Hierarchical KPI Extraction from Earnings Filings

This repository contains datasets, pre-trained models, and reproduction scripts for extracting hierarchical Key Performance Indicators (KPIs) from financial earnings filings.

## Datasets
- **[HiFi-KPI](https://huggingface.co/datasets/AAU-NLP/HiFi-KPI)**
- **[HiFi-KPI Lite](https://huggingface.co/datasets/AAU-NLP/hifi-kpi-lite)**

## Models
Available on Hugging Face:

### BERT-Base
- [Sequence Labelling (1000 most common)](https://huggingface.co/AAU-NLP/BERT-SL1000)
- [Sequence Labelling: n=1 presentation (1000 most common)](https://huggingface.co/AAU-NLP/Pre-BERT-SL1000)
- [Sequence Labelling: n=1 calculation (1000 most common)](https://huggingface.co/AAU-NLP/Cal-BERT-SL1000)
- [Sequence Labelling: HiFi-KPI Lite](https://huggingface.co/AAU-NLP/Lite-BERT-SL)

### FLANG
- [Sequence Labelling (1000 most common)](https://huggingface.co/AAU-NLP/FLANG-SL1000)
- [Sequence Labelling: n=1 presentation (1000 most common)](https://huggingface.co/AAU-NLP/Pre-FLANG-BERT-SL1000)
- [Sequence Labelling: n=1 calculation (1000 most common)](https://huggingface.co/AAU-NLP/Cal-FLANG-BERT-SL1000)
- [Sequence Labelling: HiFi-KPI Lite](https://huggingface.co/AAU-NLP/Lite-FLANG-SL)

## Setup & Installation
We use [`uv`](https://github.com/astral-sh/uv) for dependency management. Install the required dependencies by running:

```bash
uv sync
```

## Usage
### Generating Data
You can use the `main.py` template script to generate a JSON file with your preferred granularity of the iXBRL tags:

```bash
python3 main.py --taxonomy "calculation" --iterations 1
```

## Reproduction Code

### 1. TC Classification
Run the TC classification on the different levels of granularity:

```bash
uv run Training/TC.py
```

Produce the evaluation plots shown in the paper:

```bash
uv run TC_reports.py
```

### 2. Sequence Labelling
To reproduce the sequence labelling performance, run the test script. You can adjust the parameters at the top of the `test_SL.py` file to match your specific testing requirements. 

> **Note:** The `Granularity1_test.json` file required for this step is generated using the `main.py` script mentioned in the Usage section.

```bash
uv run test_SL.py
```