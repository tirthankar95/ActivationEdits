# CVEFixes Data

Scripts and data files for downloading, enriching, and splitting the CVEFixes dataset for use in safety alignment experiments.

## Files

| File | Description |
|---|---|
| `CVEFixes.csv` | Raw CVE/CWE records with code snippets and language labels (downloaded from Kaggle) |
| `CWE_descriptions.csv` | CWE ID → description mapping used as RAG knowledge base |
| `language_split.py` | Splits `CVEFixes.csv` into sample subsets using configurable strategy |
| `cwe_label.py` | RAG pipeline that labels code snippets with CWE IDs using a local LLM |
| `download_cwe_kaggle.sh` | Downloads `CVEFixes.csv` from Kaggle |
| `download_CWE_desc.py` | Downloads CWE descriptions from HuggingFace and saves to `CWE_descriptions.csv` |
| `language_distribution.png` | Bar chart of language frequency in the dataset |

## Setup

Download the raw dataset from Kaggle:

```bash
bash data/CVEFixes/download_cwe_kaggle.sh
```

Download CWE descriptions:

```bash
uv run python data/CVEFixes/download_CWE_desc.py
```

## Splitting the Dataset

Splitting is controlled by [config/lang_split.yaml](../../config/lang_split.yaml).

Two strategies are available:

**`random`** — samples 20% of the full dataset at random. Each run can produce a distinct sample identified by `sample_id`. Output is written to `data/CVEFixes/sample_random/cve_samples_<sample_id>.csv`.

**`program`** — splits the dataset by programming language. One CSV per language is written to `data/CVEFixes/sample_prog_lang/<language>.csv`.

Run with defaults (random, sample 1):

```bash
uv run python data/CVEFixes/language_split.py
```

Override from the command line:

```bash
# program split
uv run python data/CVEFixes/language_split.py strategy=program

# new random sample
uv run python data/CVEFixes/language_split.py sample_id=2
```

## CWE Labeling

`cwe_label.py` uses a RAG pipeline backed by ChromaDB and a local LLM to assign CWE labels to code snippets. It requires:

- A running local LLM server at `http://localhost:8080/v1`
- An `OPENAI_API_KEY` in `.env` (used for the LangChain client even with a local endpoint)

```bash
# First run: populate the vector store from CWE_descriptions.csv
uv run python data/CVEFixes/cwe_label.py --refresh

# Subsequent runs: skip re-insertion
uv run python data/CVEFixes/cwe_label.py
```

## Output Convention

Processed sample files must be placed under a folder named `sample_*` inside `data/CVEFixes/` so that `train_low_rank/prepare_data.py` can discover them automatically via glob.
