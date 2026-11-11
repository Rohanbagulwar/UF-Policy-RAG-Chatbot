# UF Policy RAG Chatbot

**EEE 6778 Term Project · University of Florida**

UF Policy Assistant is a retrieval-augmented question-answering system for
University of Florida policies. The current application searches locally
indexed passages, generates an answer from retrieved context, and shows
citations, excerpts, and official UF links. The updated project plan tests
whether a BGE retriever fine-tuned on student-style questions improves
retrieval and citation support compared with BM25 and pretrained BGE.

## Project deliverables

This repository is organized to support three staged submissions:

1. **Data preparation:** policy collection, scraped data, and preparation work
   are in `data/raw/Web_Scrapped_Dataset/` and
   [`notebooks/policy_scraper_and_chunking_01.ipynb`](notebooks/policy_scraper_and_chunking_01.ipynb).
   The updated [executive summary](docs/uf_policy_rag_executive_summary_updated.pdf)
   defines the current research question and comparison plan.
2. **Technical blueprint:** the architecture, dataset, interface design,
   implementation timeline, and responsible-AI considerations are described in
   the [technical blueprint](docs/uf_policy_rag_technical_blueprint.pdf).
3. **Implemented system and evaluation:** the retrieval pipeline, Streamlit
   interface, evaluation notebooks, scores, and visualizations are included in
   `src/`, `ui/`, `notebooks/`, and `results/`.

The existing code implements the pretrained BGE RAG baseline. Fine-tuning,
BM25 and hybrid retrieval, and the independent final test set are planned
experiments; they are not represented by the current evaluation workbooks.

## Requirements

- Python 3.10 or later
- An active `OPENAI_API_KEY` accepted by the UF AI gateway configured by the
  project
- Internet access for model/API calls and, when needed, downloading Python
  packages or the embedding model

## Installation and setup

Run from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
export OPENAI_API_KEY="your-current-api-key"
```

Keep the API key in the environment; do not put it in source files or notebooks.
If the key changes, restart the application in a terminal where the new
environment variable is set.

## Run the Streamlit application

```bash
streamlit run ui/streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501). Ask a policy question,
review the answer's numbered citations and excerpts, and open the linked
official policy page to verify the source. Generated answers can be incomplete
or incorrect; the official policy page is the source of truth.

The app loads the existing local ChromaDB data from `data/chroma_db/` and uses
the `policies` collection. Do not remove or replace this directory unless you
intend to rebuild the index.

### Run with Docker

Build the image from the repository root and provide the API key when running
the container:

```bash
docker build -t uf-policy-assistant .
docker run --rm -p 8501:8501 \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  uf-policy-assistant
```

Open [http://localhost:8501](http://localhost:8501). The image includes the
local ChromaDB index. The embedding model is downloaded when the app initializes.

## App screenshots

The app also includes a collapsible **Project screenshots** gallery with the
submitted interface, citation, and official-policy-page screenshots from
[`docs/visuals/`](docs/visuals/).

## Run the notebooks

Start Jupyter from the repository root so notebooks can locate project data and
modules:

```bash
jupyter lab
```

Run [`notebooks/setup.ipynb`](notebooks/setup.ipynb) first. It checks the
environment, loads and summarizes the prepared policy workbook, and verifies
the local ChromaDB collection without making an API call.
In VS Code, select the Python interpreter from this project's `.venv` as the
notebook kernel; the setup notebook reports when application dependencies are
missing from the currently selected kernel.

Open and run the workflow notebooks in this order as needed:

1. [`notebooks/setup.ipynb`](notebooks/setup.ipynb) — verify dependencies,
   load the prepared dataset, and check the local ChromaDB index.
2. [`notebooks/policy_scraper_and_chunking_01.ipynb`](notebooks/policy_scraper_and_chunking_01.ipynb) —
   collection, extraction, and chunking.
3. [`notebooks/Rag_workflow_02.ipynb`](notebooks/Rag_workflow_02.ipynb) —
   embedding and vector-store workflow.
4. [`notebooks/retrival_pipeline_03.ipynb`](notebooks/retrival_pipeline_03.ipynb) —
   retrieval and generation experiments.
5. [`notebooks/retrive_answers_on_golden_dataset_04.ipynb`](notebooks/retrive_answers_on_golden_dataset_04.ipynb)
   and [`notebooks/RAG_chatbot_different_llm_answers_05.ipynb`](notebooks/RAG_chatbot_different_llm_answers_05.ipynb) —
   answer generation and model comparisons. The golden-dataset notebook asks
   for the API key using hidden input, runs the configured model, and saves
   resumable answers under `results/Accuracy_scores/`.
6. [`notebooks/RAG_accuracy_score_comparison_06.ipynb`](notebooks/RAG_accuracy_score_comparison_06.ipynb) —
   evaluation and plots.

The accuracy notebook installs its NLTK, ROUGE, and plotting dependencies into
the active notebook kernel (not a separate `pip3 --user` Python installation).
For a reproducible environment, install `requirements.txt` first and select
that same environment as the notebook kernel.

## Dataset

The project uses public policy pages from the University of Florida's Policy
and Regulation Hub. Scraped policy content and metadata are stored under
`data/raw/Web_Scrapped_Dataset/`. The `policies_data.xlsx` workbook currently
contains 1,887 text chunks across 189 policy titles. It records title,
category, type, official URL, chunk number, and text; JSON versions are in the
same directory. The persistent ChromaDB index is stored locally in
`data/chroma_db/`. Evaluation questions and generated-answer results are in
the workbooks under `results/`. The planned study will audit coverage of all
publicly accessible hub documents and create policy-level train/development/
test splits and an independently reviewed final question set.

Data is text, not personal records. Retrieved passages are sent with a user's
question to the configured language-model API to generate an answer. Do not
enter private or sensitive information. Policy pages can change, so validate
answers against the official source links.

## Project layout

```text
data/
  raw/Web_Scrapped_Dataset/  Scraped policy data and source workbook
  chroma_db/                 Persistent ChromaDB index
docs/
  uf_policy_rag_executive_summary_updated.pdf  Current research plan
  uf_policy_rag_technical_blueprint.pdf
  visuals/                   App, citation, and source-page screenshots
notebooks/                   Data preparation, retrieval, and evaluation
  setup.ipynb                Environment and prepared-data verification
results/
  Accuracy_scores/           Evaluation workbooks
  visualizations/            Evaluation plots
src/
  config.py                  Model, database, and API configuration
  retrival_pipeline.py       Retrieval pipeline entry point
  clients/                   Language-model API client
  models/                    Embedding model wrapper
  services/                  Retrieval, prompts, context, and source lookup
  vectorstore/               ChromaDB manager
ui/
  streamlit_app.py           Streamlit chat interface
requirements.txt              Python dependencies
```

## Author

- **Name:** Rohan Bagulwar
- **Contact:** Via the EEE 6778 course submission channel (Rohan Bagulwar)
