# UF Policy RAG Chatbot

The UF Policy RAG Chatbot helps users find relevant University of Florida
policy passages with plain-language questions. The Streamlit app displays a
generated answer, inline citations, retrieved excerpts, and links to official
policy pages.

Repository: https://github.com/Rohanbagulwar/UF-Policy-RAG-Chatbot

## Current implementation

- Policy snapshot with 1,887 text chunks across 189 policy titles.
- Pretrained `BAAI/bge-base-en-v1.5` embeddings and the persistent ChromaDB
  `policies` collection.
- Answer generation with `gpt-oss-20b` through the UF AI gateway.
- Streamlit chat with adjustable passage count and visible evidence.
- An optional fine-tuned BGE synthetic-trial profile when its local checkpoint
  and matching Chroma index are available. It is experimental and has not been
  validated on an independent human-authored test set.
- A 51-question exploratory evaluation using BLEU, ROUGE, and METEOR.

## Set up the environment

The project requires Python 3.11 or later and uses `uv` to install the pinned
dependencies declared in `pyproject.toml` and `uv.lock`.

```bash
git clone https://github.com/Rohanbagulwar/UF-Policy-RAG-Chatbot.git
cd UF-Policy-RAG-Chatbot
uv sync
cp .env.example .env
```

Set `OPENAI_API_KEY` in the project-root `.env` file. The application reads
this key for the UF AI gateway. `.env` is ignored by Git; keep credentials out
of source files and notebooks.

## Verify the data and run the app

1. Open [`notebooks/setup.ipynb`](notebooks/setup.ipynb) and run its cells with
   the project environment. The notebook loads the prepared workbook, previews
   and summarizes its records, and checks the ChromaDB collection.
2. Confirm that the application uses the same persistent ChromaDB directory
   (`data/chroma_db/`) and the `policies` collection. The prepared workbook is
   `data/raw/Web_Scrapped_Dataset/policies_data.xlsx`; the expected collection
   has 1,887 passages and uses the pretrained BGE-base encoder.
3. Run Streamlit from the repository root:

   ```bash
   uv run streamlit run ui/streamlit_app.py
   ```

4. Open the local URL printed by Streamlit. Ask about an indexed policy and
   adjust the passage count. Check that the answer, numbered citations,
   retrieved excerpts, and official source links display.

## Reproduce the preliminary evaluation

Run [`notebooks/RAG_accuracy_score_comparison_06.ipynb`](notebooks/RAG_accuracy_score_comparison_06.ipynb)
using its reference-answer dataset for the 51-question exploratory evaluation.
Record the corpus version, retrieval settings, generation model, and commit.
Export the generated answers and BLEU, ROUGE, and METEOR outputs. Include the
ROUGE variant, score scale, aggregation method, and any failed or skipped runs.

These three metrics describe answer-text overlap. Retrieval Recall@5, mean
reciprocal rank, claim-level citation support, and refusal behavior will be
measured on a separate held-out evaluation set.

## Known issues and next steps

- Add collection provenance, missing-value and duplicate checks, a
  passage-length plot, and a small collection example to the setup notebook.
- Audit and freeze the corpus. Build policy-based splits, reviewed training
  pairs, and an independently reviewed final test set.
- Compare BM25, pretrained BGE-small, fine-tuned BGE-small, and hybrid
  retrieval.
- Add a retrieval-method selector and an explicit insufficient-evidence
  response to Streamlit.
- Official UF policy pages remain the authority for policy decisions. Omit
  private details when using the chat.

## Author and contact

Rohan Bagulwar · M.S. in Applied Data Science · University of Florida

Contact: https://github.com/Rohanbagulwar
