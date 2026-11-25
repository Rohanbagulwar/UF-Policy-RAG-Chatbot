# UF Policy RAG Fine-Tuning Experiment: Detailed Guide

This guide explains the fine-tuning work in this repository: why it was added,
what each file does, what the generated artifacts mean, which problems came up,
how retrieval is measured, what has been run, and what remains to be done.

> **Current status:** The original policy chatbot is still using its pretrained
> BGE model and original ChromaDB collection. A separate BGE pilot checkpoint
> has been trained and compared on 23 assistant-reviewed, AI-generated
> questions. Those scores are a synthetic diagnostic, **not** an independent
> human test. A separate 1,887-passage synthetic-trial Chroma collection has
> been built and is available through the Streamlit model selector. The
> pretrained baseline remains the default.

## 1. Why this experiment exists

The project is a retrieval-augmented generation (RAG) chatbot for University of
Florida policies. At a high level, the existing application:

1. receives a user's policy question;
2. searches the local policy passage index;
3. gives retrieved passages to a language model;
4. returns an answer with citations to the official policy source.

The experiment asks a narrower question:

> Can adapting the existing BGE text-embedding model to UF-policy question /
> passage pairs improve the retriever's ability to rank the correct passage near
> the top?

This is **embedding-model fine-tuning**, not fine-tuning the answer-writing
language model. The intended training pair is:

```text
student-style question  ->  the real policy passage that answers it
```

The model learns to place the question vector closer to its matching passage
vector than to other passages. The real policy passages remain the searchable
documents. Synthetic questions and answers are training or evaluation
annotations; they are not added as policy text and are not official UF advice.

### The two separate paths

```text
Existing application (unchanged)
User question -> pretrained BGE -> original ChromaDB "policies"
              -> retrieved UF passages -> answer model -> cited response

Fine-tuning experiment (isolated)
Policy workbook -> validated manifest -> policy-level splits
                 -> synthetic question/answer/quote drafts
                 -> review -> BGE contrastive training -> pilot checkpoint
                 -> retrieval comparison -> synthetic trial index
                 -> (future) independent-test index and app integration
```

The separation is intentional. Training, evaluation, and indexing must not
silently replace the working app's model or database.

The project's older golden answer-comparison workbook has 51 rows and 49
unique questions. It compares generated answer text; it is not a retrieval
Recall/MRR/nDCG evaluation. Those previously exposed questions are excluded
from synthetic generation and rejected from the new test paths, so the new
retrieval scores are not a restatement of the old answer comparison.

### How the work progressed

1. **Trace the existing baseline first.** The app's existing index is the
   `policies` collection in `data/chroma_db`, and the configured embedding
   model is `BAAI/bge-base-en-v1.5`. The original collection was left in place.
2. **Design a separate experiment.** The experiment was scoped to improving
   question-to-passage ranking, with real policy passages as targets and
   train/development/test splits separated by whole policy URL.
3. **Prepare the data and generation workflow.** A validated manifest,
   stable IDs, split map, and resumable synthetic QA generator were added.
   The existing 1,887 workbook chunks were used; policy content was not
   rewritten or re-indexed for the app.
4. **Resolve data-quality and approval blockers.** Generation first returned
   malformed JSON and non-verbatim quotes. Retry, resume, logging, and exact
   quote validation were added. Training then correctly refused to start while
   every QA row was a draft. At the user's direction, 199 train-split rows were
   approved with notes disclosing the lack of independent human review; the
   50 development rows remained drafts.
5. **Resolve environment incompatibilities and train a small pilot.** The
   project `.venv` needed compatible `datasets` and `accelerate` versions.
   After pinning them, the 10-step MPS pilot completed and saved `pilot_v1`.
6. **Create and evaluate a separate held-out diagnostic.** Twenty-three
   AI-generated questions tied to test-split policies were checked against
   their official source pages and exact local passages. Because they are
   synthetic, they were not relabeled as human questions; they were copied to
   a separately named synthetic test file and evaluated in explicitly
   non-independent mode.
7. **Keep the independent test and production promotion pending.** The app
   now offers the synthetic-trial collection as an explicitly experimental
   option, while retaining the pretrained baseline as its default. No
   human-authored test has been locked and no independent-test index exists, so
   the trial must not be presented as production evidence.

## 2. Dataset and experiment design

### Source data

The source is the existing workbook:

[`data/raw/Web_Scrapped_Dataset/policies_data.xlsx`](../data/raw/Web_Scrapped_Dataset/policies_data.xlsx)

The workbook contains 1,887 non-empty policy text chunks. Each source row has a
title, category, document type, official URL, chunk number, and passage text.
The experimental manifest validates and preserves this provenance.

### Why split by policy URL?

Passages from a single policy are related. If one chunk from a policy were used
for training and another chunk from that same policy were used for evaluation,
the evaluation could benefit from policy-specific language the model had
already seen. The experiment therefore assigns each normalized official policy
URL to exactly one split; all of its chunks inherit that split.

With the recorded seed (`42`), the current split map contains:

| Split | Policies | Use |
|---|---:|---|
| `train` | 134 | Approved question/passage pairs may train the model. |
| `development` | 29 | Used for model-selection checks after examples are reviewed; never used as training pairs. |
| `test` | 29 | Reserved for an independent final test; never used for training or development. |

The experiment currently has 23 synthetic questions tied to 23 distinct
test-split policies. They are only a subset of the 29 test policies.

### What a generated example contains

A synthetic QA record includes a question, draft answer, exact supporting quote,
the positive chunk ID and policy ID, title, source URLs, the source passage,
question type, split, generator model, and review metadata. The quote is
required to occur verbatim inside the cited passage. That check proves the quote
is present; it does **not** by itself prove the answer is semantically correct
or unambiguous.

## 3. Source files and their responsibilities

All executable experiment modules are under
[`src/finetuning/`](../src/finetuning/). They are separate from the chatbot's
retrieval and UI modules.

| File | High-level responsibility |
|---|---|
| [`common.py`](../src/finetuning/common.py) | Defines repository/data/artifact paths, the baseline model ID and revision, SHA-256 helpers, stable IDs, strict JSONL reading/writing, and append-as-you-go JSONL saving. |
| [`manifest.py`](../src/finetuning/manifest.py) | Reads the policy workbook, validates that source links are HTTPS pages on `policy.ufl.edu`, rejects invalid/duplicate rows, creates stable policy/chunk IDs, assigns whole policies to train/development/test, and writes manifest metadata and hashes. |
| [`qa.py`](../src/finetuning/qa.py) | Selects train/development passages, asks the configured UF AI gateway for synthetic question/answer/quote drafts, validates fields and exact quote membership, detects duplicate questions, retries malformed responses or incorrect quotes, resumes completed work, and appends failures to a log. |
| [`train.py`](../src/finetuning/train.py) | Loads only approved examples belonging to train-split policies, builds question/passage pairs, and fine-tunes the baseline SentenceTransformer with `MultipleNegativesRankingLoss`. It writes a new checkpoint and metadata without overwriting an existing checkpoint. |
| [`evaluate.py`](../src/finetuning/evaluate.py) | Validates test provenance, ranks all manifest passages for each query, calculates retrieval metrics, supports approved development evaluation and a clearly labeled synthetic diagnostic, and locks an independent human test by hashes. |
| [`index.py`](../src/finetuning/index.py) | Builds a **separate** Chroma collection from real manifest passages and a fine-tuned checkpoint. By default it requires a locked independent-human evaluation; an explicit trial-only option permits the labeled synthetic report. It checks checkpoint/index compatibility and refuses the baseline database path, baseline collection name, or an existing target. |
| [`test_workflow.py`](../src/finetuning/test_workflow.py) | Focused tests for manifest validation, split isolation, QA validation/retries/resume, training selection and publication, ranking metrics, test locking, synthetic-test labels, and index compatibility/publication. |
| [`__init__.py`](../src/finetuning/__init__.py) | Makes the directory importable as the `src.finetuning` Python package. |
| [`finetuning_workflow.ipynb`](../src/finetuning/finetuning_workflow.ipynb) | Guided notebook wrapper around the modules. Its generation and training switches default to `False`, so opening/running the notebook does not unexpectedly make API calls or start training. |
| [`README.md`](../src/finetuning/README.md) | Concise command-oriented workflow and isolation notes. This guide is the longer explanation. |

The project root [`requirements.txt`](../requirements.txt) now pins
`datasets==4.4.1` and `accelerate==1.10.1`, which were needed for the installed
Sentence Transformers training API and its Accelerate integration.

## 4. Generated artifacts and what they mean

The generated files are under
[`src/finetuning/artifacts/`](../src/finetuning/artifacts/). The artifact
directory's `.gitignore` only allows the manifest, policy split map, and
manifest metadata to be tracked by Git. QA rows, reports, review files, and
model weights are ignored; keep a backup if they are important to retain.

| Artifact | Contents and current meaning |
|---|---|
| `manifest.v1.jsonl` | 1,887 real workbook passages with stable chunk IDs, policy IDs, normalized and original URLs, title, category, type, chunk number, and text. |
| `manifest_metadata.v1.json` | Source workbook path/hash, chunking provenance (`RecursiveCharacterTextSplitter`, 1,000-character chunks, 100-character overlap), counts, split seed/counts, and manifest/split hashes. Current metadata reports 1,887 accepted chunks from 192 URL-identified policies and no rejected rows. |
| `policy_splits.v1.jsonl` | One row per policy URL, recording `train`, `development`, or `test`; current counts are 134/29/29. |
| `qa_drafts.v1.jsonl` | 249 generated examples: 199 train-split rows marked `approved` at the user's direction and 50 development rows still `draft`. The approved-row notes explicitly say that automated metadata/exact-quote checks passed but the assistant did not independently verify every interpretation against the live policy. These rows remain synthetic. |
| `qa_failures.v1.jsonl` | Append-only record of failed generation attempts. It currently has 39 records: 25 `JSONDecodeError` entries and 14 quote-validation `ValueError` entries. Of the JSON errors, 23 say `Expecting value` at the first character, one reports a malformed delimiter, and one an unterminated string. Because the response body is not saved, the first group cannot be distinguished reliably as blank responses versus other non-JSON output. Four records concern the same chunk; there are 36 distinct failed chunk IDs overall. The log accumulates attempts across runs, so 39 does not mean 39 unique policies or 39 drafts were lost. |
| `human_test_candidates.v1.jsonl` | The 23 generated test-policy questions reviewed against their linked official sources and local source passages. Each is `review_status: "approved"` for assistant source review, `reviewed_by: "AI assistant"`, and `synthetic: true`. They are not human-authored. One question was narrowed to ask specifically which catalog is listed first so the question matches its supporting passage. |
| `assistant_reviewed_synthetic_test.v1.jsonl` | Copy used by the synthetic diagnostic. It has the same 23 records and preserves their synthetic provenance. It is intentionally not named `human_test.v1.jsonl`. |
| `assistant_reviewed_synthetic_evaluation.v1.json` | Comparison report for the pretrained and pilot models on the 23 synthetic queries. It explicitly contains `evaluation_type: "assistant_reviewed_synthetic_questions"` and `independent_human_test: false`. |
| `chroma_finetuned_synthetic_v1/` | **Built for the ongoing trial.** It contains the same 1,887 real manifest passages, re-embedded with `pilot_v1`, in collection `policies_finetuned_synthetic`. Metadata labels the evaluation as synthetic and `independent_human_test: false`. It is not a production index. |
| `checkpoints/pilot_v1/` | Locally saved fine-tuned SentenceTransformer files (weights, model configuration, tokenizer, pooling configuration, and `training_metadata.json`). This is the pilot checkpoint, not a deployed app model. |
| `locked_test.v1.json` / `human_test.v1.jsonl` | **Not created.** There is no independent human test set locked yet. |
| `development_evaluation.v1.json` | **Not created.** The 50 development QA examples are still drafts, so there are no approved development queries to evaluate. |
| `retrieval_evaluation.v1.json` | **Not created.** This is the report expected from the locked independent test and a prerequisite to the default independent-test index build. The separate synthetic-trial build uses `assistant_reviewed_synthetic_evaluation.v1.json`. |
| `chroma_finetuned_v1/` | **Not created.** This is the default destination for the independent-human-test index, not the synthetic-trial collection above. |

### Pilot checkpoint metadata

The saved `pilot_v1` metadata records:

- Base model: `BAAI/bge-base-en-v1.5`
- Base revision: `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`
- Embedding dimension: 768
- Loss: `MultipleNegativesRankingLoss`
- Device: MPS (Apple Silicon)
- Epochs: 1; batch size: 8; learning rate: `2e-5`; seed: 42
- Maximum training steps: 10
- Approved train pairs: 199
- Recorded training duration: 15.24 seconds
- SHA-256 fingerprints of the manifest, split map, and QA training file

The training run transcript reported a loss of approximately `0.184`. That is
the training objective's loss for the pilot run, not retrieval accuracy or
evidence that the chatbot answers improved; it is not included in the saved
checkpoint metadata.

The hashes make the checkpoint reproducible/auditable against the data it used.
If the QA file, manifest, or split map changes later, evaluation rejects the
checkpoint instead of silently comparing a model trained on different data.
The training command also refuses to overwrite an existing checkpoint; use a
new checkpoint name for another run.

## 5. End-to-end workflow and commands

Run commands from the repository root with the project's `.venv` active. In
this workspace, the environment's Python is `.venv/bin/python`.

### Step 1: Build or rebuild the manifest

```bash
python -m src.finetuning.manifest
```

This reads the source workbook and writes the manifest, policy split map, and
metadata. Rebuilding changes hashes if source data, split assignments, or
serialization changes; existing checkpoints then no longer match.

### Step 2: Generate question/answer drafts

```bash
export OPENAI_API_KEY="your-current-key"
python -m src.finetuning.qa --count 250 --workers 2
```

The default generator model is `gpt-oss-20b`; `--model` can override it. The
configured gateway can be changed with `UF_AI_BASE_URL`. Generation is
network-bound, uses two concurrent requests by default, and supports 1–8
workers. For `--count 250`, the current selector targets approximately 200
train and 50 development passages (80/20); it does not select test policies.
Each request can take time because it calls the external model; the client
timeout is 45 seconds and it permits HTTP-level retries. Generation requests
JSON and allows up to 1,200 output tokens, retries empty/malformed responses
with a larger 2,000-token cap, and separately retries an off-passage quote
once with a stricter verbatim-copy instruction.

Successful records are appended as they finish, so interruption does not
discard already-saved drafts. Re-running the same target skips completed
passages and retries failed ones. The selected source passages come from
train/development policies only; test policies are not used to generate
training data.

Before training, inspect each row's question, draft answer, exact quote, and
source URL. Confirm that the answer is supported, not broader than the quote,
and still matches the official page. A quote-membership check is necessary but
not sufficient for semantic approval.

### Step 3: Approve training examples and train

Training only uses rows that are `approved` and belong to `train` policies.
Approved development examples are excluded. It also keeps one positive
passage per training example to avoid using duplicate positives as in-batch
negative examples.

The normal pilot command is:

```bash
python -m src.finetuning.train --max-steps 10
```

For this experiment that command has already been run successfully. It saved
`pilot_v1`; rerunning with the same default checkpoint path will stop with
`FileExistsError`. To make a new pilot, use another path, for example:

```bash
python -m src.finetuning.train \
  --max-steps 10 \
  --checkpoint src/finetuning/artifacts/checkpoints/pilot_v2
```

With the default batch size of 8, enough distinct approved train passages must
be available to create at least one full batch. A 10-step pilot needs roughly
80 approved pairs to provide 10 full batches. More approved data can improve
the opportunity to learn but does not guarantee better retrieval.

### Step 4: Evaluate development data for model selection

```bash
python -m src.finetuning.evaluate \
  --development \
  --checkpoint src/finetuning/artifacts/checkpoints/pilot_v1
```

This command requires approved development questions. The 50 current
development rows remain `draft`, so this step is not yet runnable with the
current data. Reviewing and approving development questions is a separate task;
they must not be added to training pairs.

### Step 5: Run the available synthetic diagnostic

```bash
python -m src.finetuning.evaluate \
  --synthetic-test \
  --test src/finetuning/artifacts/assistant_reviewed_synthetic_test.v1.jsonl \
  --checkpoint src/finetuning/artifacts/checkpoints/pilot_v1
```

This command has been run successfully (about 1.5 minutes in the recorded
environment, including passage encoding for both models). It checks that each row is approved
and explicitly synthetic, has required fields, has a unique question, points
to the matching held-out test-policy chunk and source URL, and has an exact
quote in that source passage. It also checks for overlap with questions in the
previously exposed golden answer workbook.

This is useful for debugging the retrieval pipeline and comparing the
checkpoint on a provisional set. It is **not** the independent final test:
the questions are AI-generated and assistant-reviewed.

For the ongoing engineering trial only, the synthetic report can also be used
to build a clearly labeled separate collection:

```bash
python -m src.finetuning.index \
  --checkpoint src/finetuning/artifacts/checkpoints/pilot_v1 \
  --allow-synthetic-evaluation
```

This writes `artifacts/chroma_finetuned_synthetic_v1` with collection
`policies_finetuned_synthetic`. It re-embeds all manifest passages with the
fine-tuned checkpoint. Its Chroma metadata records that the evaluation was
synthetic and not an independent human test. This collection has now been
built and smoke-tested, and the Streamlit app now offers it as an experimental
retrieval option. The baseline model and collection remain the default. The
opt-in flag does not weaken the default independent-test gate.

### Step 6: Prepare and lock the independent human test (still pending)

The final test must consist of questions authored/reviewed by a human who
checks the answer and positive passage against the official policy page. Use
only `test`-split policies. Each JSONL row must include non-empty
`question`, `positive_chunk_id`, `policy_id`, `question_type`, and
`reviewed_by`; do not set `synthetic` to false for generated questions.

After a genuine human test file exists:

```bash
python -m src.finetuning.evaluate --lock-test
python -m src.finetuning.evaluate \
  --checkpoint src/finetuning/artifacts/checkpoints/pilot_v1
```

The lock stores hashes for the test rows, manifest, and split map. Editing any
of these invalidates the lock. The test validator rejects synthetic questions,
duplicate questions, chunks outside held-out test policies, and questions
already present in the exposed golden dataset. The fine-tuned report also
checks the checkpoint's corpus, split, QA-data, and base-model fingerprints.
Do not tune repeatedly against the final test; use development data for model
selection.

### Step 7: Build the independent-test index (still pending)

Only after a valid locked-test comparison and a non-regressing retrieval result:

```bash
python -m src.finetuning.index \
  --checkpoint src/finetuning/artifacts/checkpoints/pilot_v1
```

The default output is
`src/finetuning/artifacts/chroma_finetuned_v1`, collection
`policies_finetuned`. The builder indexes real passages only, requires the
report from the locked independent test, checks checkpoint fingerprints and
metadata, and publishes from a temporary directory only after all chunks have
been indexed. It refuses to modify the original `data/chroma_db` baseline,
reuse the original `policies` collection name, or overwrite an existing
destination. The synthetic collection built in Step 5 is separate and is not
a substitute for this independent-test index.

## 6. How retrieval “accuracy” is measured

The evaluator embeds every manifest passage and each test question with the
same model, using normalized vectors. It scores query/passage similarity,
sorts the 1,887 passages, and checks where the labeled positive chunk appears.
For each query, there is one expected positive chunk. A positive chunk outside
the top 10 receives no reciprocal-rank or nDCG credit.

| Metric | Meaning |
|---|---|
| Recall@1 | Fraction of queries whose expected passage is ranked first. |
| Recall@3 | Fraction whose expected passage is in positions 1–3. |
| Recall@5 | Fraction whose expected passage is in positions 1–5. |
| MRR@10 | Mean reciprocal rank of the expected passage if it appears in the top 10; otherwise zero. Rank 1 contributes 1, rank 2 contributes 1/2, and so on. |
| nDCG@10 | Discounted ranking score for the single expected passage; earlier ranks receive more credit, and a passage outside the top 10 receives zero. |

These are **retrieval metrics**, not answer-generation accuracy. They do not
measure whether an answer-writing LLM gives correct prose, whether a citation
is displayed correctly in the UI, or whether every policy interpretation is
complete. Those require separate end-to-end and human correctness tests.

### Current 23-question synthetic comparison

| Metric | Pretrained BGE | Fine-tuned pilot | Difference |
|---|---:|---:|---:|
| Recall@1 | 0.6957 (16/23) | 0.7826 (18/23) | +0.0869 |
| Recall@3 | 0.7826 (18/23) | 0.9130 (21/23) | +0.1304 |
| Recall@5 | 0.9130 (21/23) | 0.9130 (21/23) | 0.0000 |
| MRR@10 | 0.7739 | 0.8377 | +0.0638 |
| nDCG@10 | 0.8174 | 0.8661 | +0.0487 |

The pilot ranked the expected passage first more often on this set, while both
models placed it in the top five for 21 of 23 questions. This is encouraging
debugging evidence, but it does not prove a general retrieval improvement:
the sample is small, generated by AI, and reviewed by an AI assistant. A human
independent test is still needed before claiming an experimental gain or
building the fine-tuned application index.

## 7. Problems encountered and what changed

### “No human-approved train pairs”

The training code intentionally rejects unreviewed examples. At the time,
there were 249 generated rows but they were still drafts, so `training_pairs()`
correctly returned no eligible examples. The user then authorized marking the
199 train-split rows approved; the 50 development rows stayed drafts. The
recorded notes state that those training examples were not independently
verified against live policy pages by the assistant. Thus the pipeline ran,
but this approval history should not be confused with a completed human audit.

### Empty or malformed generation responses

The failure log contains responses that could not be parsed as JSON, including
`Expecting value`, `Expecting ',' delimiter`, and `Unterminated string`
errors. The stored log does not include raw response bodies, so it cannot tell
whether an `Expecting value` response was empty, plain text, or another
non-JSON body. The generator was made more resilient by requesting JSON mode,
trying a larger output-token cap after malformed/empty output, retrying
HTTP-level transient errors through the client, saving each successful row
immediately, and resuming already-completed chunks on rerun.

### Quote not copied exactly

The model sometimes paraphrased or altered punctuation instead of copying a
contiguous passage substring. The generator validates the quote against the
source chunk and retries the affected chunk once with a stricter verbatim-copy
instruction. If the retry still fails, the chunk is logged and skipped; this
is why a run can finish with 249 drafts rather than the target of 250.

The attached/current failure log has 14 quote-validation records. The same
chunk ID appears multiple times because failed chunks were retried across
separate runs; inspect the log as attempt history, not as a count of unique
policies or final dataset size.

### Ctrl+C seemed to take a long time

Generation uses a thread pool for network calls. The code cancels queued
futures and requests non-waiting executor shutdown on interruption, but a
request already in progress cannot be forcibly stopped by `Future.cancel()`.
The HTTP client also has a 45-second timeout and retry behavior. Consequently,
an interrupted process may take time to finish outstanding network activity.
Because rows are appended incrementally, already-finished rows remain saved.

### `datasets` reported missing even though it had been installed

The install and training command were using different Python environments.
Installing a package into a system/user Python does not install it into the
project's `.venv`. The dependency was pinned as `datasets==4.4.1` and
installed/verified in the project environment. Run the module from the
repository root with the `.venv` interpreter/kernel selected.

### `Accelerator.unwrap_model()` rejected `keep_torch_compile`

This was a version-compatibility problem between the training stack and
Accelerate, rather than a bad dataset. The project dependency list now pins
`accelerate==1.10.1` with `datasets==4.4.1`; the training call was verified in
the project environment and the pilot completed.

### Retrieval evaluation emitted NumPy matrix warnings

During the comparison, the matrix multiplication path emitted divide-by-zero,
overflow, and invalid-operation warnings even though repeated inspection found
finite normalized model vectors and finite results. The scorer was changed to
convert vectors to `float64`, explicitly reject non-finite embeddings/scores,
and compute pairwise dot products with `numpy.einsum`. The final comparison
completed without those warnings and returned the same reported metrics.

### Why the 23 generated candidates were not put in `human_test.v1.jsonl`

The user asked for the generated questions to be marked non-synthetic. That
would falsely describe their origin. The candidates remain
`synthetic: true`, identify the reviewer as an AI assistant, and include a note
that they are not independent human review. A separate
`assistant_reviewed_synthetic_test.v1.jsonl` and explicit evaluator mode were
created instead. The human-test lock still rejects synthetic data.

## 8. Test coverage and current evidence

The focused command is:

```bash
python -m unittest src.finetuning.test_workflow
```

The latest run passed 16 tests. They cover:

- workbook row validation, provenance, and stable splitting;
- isolation of whole policies across splits;
- QA row checks, generated output retries, failure logging, concurrency, and resume;
- exclusion of development examples from training;
- checkpoint publication only after successful training;
- index compatibility, safe separate publication, and model reuse by the retriever;
- metric calculations and non-finite-vector rejection;
- lock invalidation when the human test or split map changes;
- synthetic test quote/source/split validation and rejection by the human-test lock.

`git diff --check` also passed after the code changes. The actual synthetic
comparison completed for both pretrained BGE and `pilot_v1` and wrote the
report listed above.

## 9. Practical next steps

1. **Human-audit the 199 training examples** if the experiment needs stronger
   training-data quality claims. Their recorded approval was user-authorized,
   not an independent page-by-page human audit. If any row changes, use a new
   checkpoint path and retrain because checkpoint QA hashes will no longer
   match.
2. **Review and approve development questions** from development policies
   before using `--development` to select a model. Keep test-policy questions
   out of training and model selection.
3. **Create a genuinely human-authored test set** using held-out test policies.
   Do not relabel the synthetic candidate set as human-authored.
4. **Lock and evaluate that independent test once.** Report metrics with the
   test size and provenance, and do not tune against it.
5. **Build the independent-test Chroma collection only after that comparison
   passes.** A separately labeled synthetic-trial collection exists now. Then
   compare answers, citations, latency, and failure behavior before considering
   app integration.

## 10. Glossary

- **Chunk / passage:** A segment of an official policy page stored as a
  searchable document.
- **Embedding:** A numeric vector representing text for similarity search.
- **Positive passage:** The passage labeled as the correct source for a
  question.
- **Synthetic example:** A question/answer written by a model rather than a
  human; its answer remains a draft until reviewed.
- **Policy-level split:** Assignment of every passage from one policy URL to a
  single train, development, or test group.
- **Contrastive training:** Adjusting embeddings so matched question/passage
  pairs rank closer than unmatched pairs.
- **Checkpoint:** A saved copy of model weights and training metadata.
- **Independent test:** A held-out evaluation authored/reviewed by a human,
  kept out of training and tuning.
