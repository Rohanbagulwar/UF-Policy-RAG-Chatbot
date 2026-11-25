"""Focused checks for isolated corpus, review, ranking, and index guards."""
from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from threading import Lock
from unittest.mock import patch

import chromadb
import numpy as np
import pandas as pd
from sentence_transformers import InputExample

from ..config import ChromaDBConfig
from ..services.context_builder_service import ContextBuilder
from ..vectorstore.chroma_manager import ChromaDBManager
from .common import read_jsonl, sha256_file, write_jsonl
from .evaluate import (
    LOCK_FILE,
    evaluate_development,
    evaluate_synthetic_test,
    lock_test,
    metric_at_rank,
    rank_model,
    validate_locked_test,
    validate_synthetic_test,
)
from .index import FineTunedRetriever, build as build_index, compatible, model_fingerprint
from .manifest import parse_workbook, split_policies
from .qa import _generate_one, generate, select_pilot, validate_example
from .train import train, training_pairs


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)

    def test_chroma_manager_does_not_create_required_missing_collection(self):
        manager = ChromaDBManager(
            ChromaDBConfig(
                persist_directory=str(self.path / "missing_index"),
                collection_name="required_collection",
                require_existing_collection=True,
            )
        )
        with self.assertRaisesRegex(FileNotFoundError, "required_collection"):
            manager.initialize()

    def test_context_builder_accepts_trial_metadata_without_type(self):
        result = {
            "documents": [["Policy excerpt."]],
            "metadatas": [[{"title": "Test Policy", "category": "Academic"}]],
            "distances": [[0.1]],
        }
        self.assertIn("Type: Policy", ContextBuilder.prepare_context(result))
        self.assertEqual(ContextBuilder.extract_source_documents(result)[0]["type"], "Policy")

    def test_manifest_rejects_bad_rows_and_preserves_provenance(self):
        rows = [dict(title="Test", category="Student Life", type="Policy", link="https://policy.ufl.edu/policy/test/", chunk_number=1, content="Real passage."),
                dict(title="Test", category="Student Life", type="Policy", link="https://policy.ufl.edu/policy/test/", chunk_number=1, content="Real passage."),
                dict(title="Bad", category="Student Life", type="Policy", link="https://example.com/policy/bad/", chunk_number=1, content="Other passage.")]
        workbook = self.path / "sample.xlsx"
        with pd.ExcelWriter(workbook) as writer:
            pd.DataFrame(rows).to_excel(writer, sheet_name="Policies", index=False)
        manifest, metadata = parse_workbook(workbook)
        self.assertEqual(len(manifest), 1)
        self.assertEqual(sum(metadata["rejected"].values()), 2)
        self.assertEqual(manifest[0]["original_source_url"], rows[0]["link"])
        self.assertEqual(manifest[0]["text"], "Real passage.")

    def test_policy_split_isolation(self):
        rows = [{"policy_id": f"policy-{i // 3}", "category": "Student Life"} for i in range(90)]
        splits = split_policies(rows, seed=42)
        self.assertEqual(len(splits), 30)
        self.assertEqual(set(splits.values()), {"train", "development", "test"})
        self.assertEqual(splits, split_policies(rows, seed=42))
        chunks = [{"chunk_id": f"c{i}", "policy_id": f"p{i}", "category": "Student Life"} for i in range(10)]
        assignments = {f"p{i}": "train" if i < 7 else "development" for i in range(10)}
        selected = select_pilot(chunks, assignments, 10, 42)
        self.assertEqual(len(selected), 9)
        self.assertEqual(sum(assignments[row["policy_id"]] == "development" for row in selected), 2)

    def test_policy_split_keeps_all_splits_with_small_categories(self):
        rows = [{"policy_id": f"p{i}", "category": f"category-{i}"} for i in range(3)]
        splits = split_policies(rows, seed=42)
        self.assertEqual(set(splits.values()), {"train", "development", "test"})

    def test_qa_validation_and_training_format(self):
        chunk = {"chunk_id": "c1", "policy_id": "p1", "title": "Test", "source_url": "https://policy.ufl.edu/policy/test/",
                 "original_source_url": "https://policy.ufl.edu/policy/test/", "text": "Students must submit the form by Friday."}
        development_chunk = {"chunk_id": "c2", "policy_id": "p2", "title": "Test two",
                             "source_url": "https://policy.ufl.edu/policy/test-two/",
                             "original_source_url": "https://policy.ufl.edu/policy/test-two/",
                             "text": "Students may submit the form online."}
        example = {"question": "When must I submit the form?", "draft_answer": "By Friday.",
                   "supporting_quote": "Students must submit the form by Friday.", "positive_chunk_id": "c1",
                   "policy_id": "p1", "title": "Test", "source_url": chunk["source_url"],
                   "original_source_url": chunk["original_source_url"], "chunk_text": chunk["text"],
                   "question_type": "student wording", "synthetic": True, "review_status": "approved",
                   "reviewed_by": "reviewer", "review_note": "Verified against official source."}
        development_example = {**example, "question": "Can I submit the form online?",
                               "draft_answer": "Yes, online.", "positive_chunk_id": "c2",
                               "supporting_quote": "Students may submit the form online.",
                               "policy_id": "p2", "title": development_chunk["title"],
                               "source_url": development_chunk["source_url"],
                               "original_source_url": development_chunk["original_source_url"],
                               "chunk_text": development_chunk["text"], "split": "development"}
        chunks = {"c1": chunk, "c2": development_chunk}
        validate_example(example, chunks, set())
        write_jsonl(self.path / "manifest.v1.jsonl", [chunk, development_chunk])
        write_jsonl(self.path / "policy_splits.v1.jsonl", [
            {"policy_id": "p1", "split": "train"}, {"policy_id": "p2", "split": "development"},
        ])
        write_jsonl(self.path / "qa_drafts.v1.jsonl", [example, development_example])
        pairs = training_pairs(self.path)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].texts, [example["question"], chunk["text"]])
        metrics = {"overall": {"recall@5": 1.0, "mrr@10": 1.0}, "queries": []}
        with patch("src.finetuning.evaluate.rank_model", return_value=metrics) as ranker:
            evaluate_development(self.path, checkpoint=None, batch_size=8)
        evaluated_questions = ranker.call_args.args[2]
        self.assertEqual([row["question"] for row in evaluated_questions],
                         [development_example["question"]])
        report = json.loads((self.path / "development_evaluation.v1.json").read_text())
        self.assertEqual(report["development_questions"], 1)
        example["supporting_quote"] = "Unsupported quote"
        with self.assertRaisesRegex(ValueError, "quote"):
            validate_example(example, chunks, set())
        example["supporting_quote"] = chunk["text"]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_example(example, chunks, {"when must i submit the form"})

    def test_qa_generation_runs_concurrently_and_resumes(self):
        out = self.path / "generation"
        out.mkdir()
        chunks = []
        splits = []
        for index in range(4):
            policy_id = f"policy-{index}"
            url = f"https://policy.ufl.edu/policy/test-{index}/"
            chunks.append({
                "chunk_id": f"chunk-{index}", "policy_id": policy_id, "source_url": url,
                "original_source_url": url, "title": f"Test policy {index}",
                "category": "Student Life", "type": "Policy", "chunk_number": 1,
                "text": f"Source rule {index}.",
            })
            splits.append({"policy_id": policy_id, "split": "development" if index == 3 else "train"})
        write_jsonl(out / "manifest.v1.jsonl", chunks)
        write_jsonl(out / "policy_splits.v1.jsonl", splits)

        lock = Lock()
        active = 0
        max_active = 0

        def create_completion(**kwargs):
            nonlocal active, max_active
            prompt = kwargs["messages"][1]["content"]
            title = prompt.split("Title: ", 1)[1].splitlines()[0]
            index = int(title.rsplit(" ", 1)[1])
            with lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.03)
            with lock:
                active -= 1
            question_templates = (
                "Which rule applies to this test policy?",
                "What requirement does the second policy describe?",
                "In the third policy, what is the stated condition?",
                "How does the final policy address this issue?",
            )
            payload = json.dumps({
                "question": question_templates[index],
                "draft_answer": f"Source rule {index}.",
                "supporting_quote": f"Source rule {index}.",
            })
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=payload))])

        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create_completion))
        )
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            with patch("openai.OpenAI", return_value=fake_client):
                result = generate(out, count=4, workers=3)
                self.assertGreaterEqual(max_active, 2)
                self.assertEqual(result["new_drafts"], 4)
                resumed = generate(out, count=4, workers=3)
        self.assertEqual(len(read_jsonl(out / "qa_drafts.v1.jsonl")), 4)
        self.assertEqual(resumed["new_drafts"], 0)

    def test_qa_generation_persists_failure_reason(self):
        out = self.path / "failed_generation"
        out.mkdir()
        chunk = {
            "chunk_id": "chunk-1", "policy_id": "policy-1",
            "source_url": "https://policy.ufl.edu/policy/test/",
            "original_source_url": "https://policy.ufl.edu/policy/test/",
            "title": "Test policy", "category": "Student Life", "type": "Policy",
            "chunk_number": 1, "text": "Source rule.",
        }
        write_jsonl(out / "manifest.v1.jsonl", [chunk])
        write_jsonl(out / "policy_splits.v1.jsonl", [{"policy_id": "policy-1", "split": "train"}])
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kwargs: SimpleNamespace(
                        choices=[SimpleNamespace(message=SimpleNamespace(content="not JSON"))]
                    )
                )
            )
        )
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            with patch("openai.OpenAI", return_value=fake_client):
                result = generate(out, count=1, workers=1)
        failure = read_jsonl(out / "qa_failures.v1.jsonl")
        self.assertEqual(result["new_drafts"], 0)
        self.assertEqual(failure[0]["error_type"], "InvalidModelResponse")
        self.assertEqual(failure[0]["chunk_id"], "chunk-1")

    def test_qa_generation_retries_empty_or_truncated_completions(self):
        requests = []
        valid_content = json.dumps({
            "question": "What rule applies to this policy?",
            "draft_answer": "The stated rule applies.",
            "supporting_quote": "The stated rule applies.",
        })
        responses = [
            SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content=""),
                    finish_reason="length",
                )],
                usage=SimpleNamespace(completion_tokens=400),
            ),
            SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content=valid_content),
                    finish_reason="stop",
                )],
            ),
        ]

        class FakeCompletions:
            def create(self, **kwargs):
                requests.append(kwargs)
                return responses.pop(0)

        client = SimpleNamespace(
            chat=SimpleNamespace(completions=FakeCompletions())
        )
        chunk = {
            "chunk_id": "chunk-1", "policy_id": "policy-1",
            "source_url": "https://policy.ufl.edu/policy/test/",
            "original_source_url": "https://policy.ufl.edu/policy/test/",
            "title": "Test policy", "text": "The stated rule applies.",
        }
        result = _generate_one(client, chunk, "direct", "test-model", "train")
        self.assertEqual(result["question"], "What rule applies to this policy?")
        self.assertEqual([request["max_tokens"] for request in requests], [1200, 2000])
        self.assertTrue(all(request["response_format"] == {"type": "json_object"} for request in requests))

    def test_generation_retries_non_verbatim_supporting_quote(self):
        out = self.path / "quote_retry"
        out.mkdir()
        chunk = {
            "chunk_id": "chunk-1", "policy_id": "policy-1",
            "source_url": "https://policy.ufl.edu/policy/test/",
            "original_source_url": "https://policy.ufl.edu/policy/test/",
            "title": "Test policy", "category": "Student Life", "type": "Policy",
            "chunk_number": 1, "text": "The exact policy sentence applies.",
        }
        write_jsonl(out / "manifest.v1.jsonl", [chunk])
        write_jsonl(out / "policy_splits.v1.jsonl", [{"policy_id": "policy-1", "split": "train"}])
        calls = []

        def completion(**kwargs):
            calls.append(kwargs)
            correction = "Your prior quote was not an exact match." in kwargs["messages"][1]["content"]
            quote = chunk["text"] if correction else "The policy sentence applies."
            content = json.dumps({
                "question": "What policy sentence applies?",
                "draft_answer": "The exact policy sentence applies.",
                "supporting_quote": quote,
            })
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content=content),
                    finish_reason="stop",
                )]
            )

        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=completion))
        )
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            with patch("openai.OpenAI", return_value=fake_client):
                result = generate(out, count=1, workers=1)
        saved = read_jsonl(out / "qa_drafts.v1.jsonl")
        self.assertEqual(result["new_drafts"], 1)
        self.assertEqual(len(calls), 2)
        self.assertEqual(saved[0]["supporting_quote"], chunk["text"])
        failure_path = out / "qa_failures.v1.jsonl"
        self.assertFalse(failure_path.exists() and read_jsonl(failure_path))

    def test_index_compatibility(self):
        checkpoint = self.path / "checkpoint"
        checkpoint.mkdir()
        (checkpoint / "model.safetensors").write_bytes(b"weights")
        from .index import model_fingerprint
        good = {"model_fingerprint": model_fingerprint(checkpoint), "embedding_dimension": 768, "manifest_sha256": "abc"}
        compatible(good, checkpoint, 768, "abc")
        with self.assertRaisesRegex(ValueError, "embedding_dimension"):
            compatible(good, checkpoint, 384, "abc")
        with self.assertRaisesRegex(ValueError, "model_fingerprint"):
            (checkpoint / "model.safetensors").write_bytes(b"changed")
            compatible(good, checkpoint, 768, "abc")

    def test_retriever_reuses_loaded_model_for_queries(self):
        checkpoint = self.path / "checkpoint"
        checkpoint.mkdir()
        (checkpoint / "model.safetensors").write_bytes(b"weights")
        metadata = {
            "model_fingerprint": model_fingerprint(checkpoint),
            "embedding_dimension": 2,
            "manifest_sha256": "abc",
        }

        class FakeModel:
            def get_sentence_embedding_dimension(self):
                return 2

            def encode(self, texts):
                return np.array([[0.1, 0.2] for _ in texts])

        class FakeCollection:
            def __init__(self):
                self.metadata = metadata
                self.calls = []

            def query(self, **kwargs):
                self.calls.append(kwargs)
                return {"ids": [["chunk"]]}

        collection = FakeCollection()
        with patch("src.finetuning.index.SentenceTransformer", return_value=FakeModel()) as loader:
            retriever = FineTunedRetriever(collection, checkpoint, "abc")
            self.assertEqual(retriever.query("question")["ids"], [["chunk"]])
            self.assertEqual(retriever.query("another question")["ids"], [["chunk"]])
        loader.assert_called_once_with(str(checkpoint))
        self.assertEqual(len(collection.calls), 2)

    def test_training_publishes_checkpoint_only_after_success(self):
        out = self.path / "training_data"
        out.mkdir()
        write_jsonl(out / "manifest.v1.jsonl", [{"chunk_id": "c1", "text": "Passage"}])
        write_jsonl(out / "policy_splits.v1.jsonl", [{"policy_id": "p1", "split": "train"}])
        write_jsonl(out / "qa_drafts.v1.jsonl", [{"question": "Question", "review_status": "approved"}])
        pairs = [
            InputExample(texts=["Question one", "Passage one"]),
            InputExample(texts=["Question two", "Passage two"]),
        ]

        class FakeModel:
            def fit(self, **kwargs):
                (Path(kwargs["output_path"]) / "model.safetensors").write_bytes(b"trained weights")

            def get_sentence_embedding_dimension(self):
                return 2

        checkpoint = self.path / "checkpoints" / "pilot"
        with patch("src.finetuning.train.training_pairs", return_value=pairs):
            with patch("src.finetuning.train.SentenceTransformer", return_value=FakeModel()):
                with patch("src.finetuning.train.losses.MultipleNegativesRankingLoss", return_value=object()):
                    metadata = train(out, checkpoint, "cpu", 1, 2, 2e-5, 42, 1)
        self.assertTrue((checkpoint / "model.safetensors").is_file())
        self.assertTrue((checkpoint / "training_metadata.json").is_file())
        self.assertEqual(metadata["approved_pairs"], 2)
        self.assertFalse(any(checkpoint.parent.glob(".pilot.train-*")))

    def test_index_build_publishes_separate_complete_collection(self):
        out = self.path / "artifacts"
        out.mkdir()
        checkpoint = self.path / "checkpoint"
        checkpoint.mkdir()
        (checkpoint / "model.safetensors").write_bytes(b"weights")
        chunks = [
            {"chunk_id": "c1", "policy_id": "p1", "title": "Policy one", "category": "Student Life",
             "source_url": "https://policy.ufl.edu/policy/one/", "text": "Passage one."},
            {"chunk_id": "c2", "policy_id": "p2", "title": "Policy two", "category": "Student Life",
             "source_url": "https://policy.ufl.edu/policy/two/", "text": "Passage two."},
        ]
        write_jsonl(out / "manifest.v1.jsonl", chunks)
        write_jsonl(out / "policy_splits.v1.jsonl", [
            {"policy_id": "p1", "split": "test"}, {"policy_id": "p2", "split": "train"},
        ])
        test = out / "human_test.v1.jsonl"
        write_jsonl(test, [{"question": "Question?", "positive_chunk_id": "c1"}])
        lock_record = {
            "schema_version": 1,
            "test_sha256": sha256_file(test),
            "manifest_sha256": sha256_file(out / "manifest.v1.jsonl"),
            "splits_sha256": sha256_file(out / "policy_splits.v1.jsonl"),
        }
        (out / LOCK_FILE).write_text(json.dumps(lock_record), encoding="utf-8")
        report = {
            "fine_tuned_model": str(checkpoint),
            "fine_tuned_fingerprint": model_fingerprint(checkpoint),
            "test_sha256": lock_record["test_sha256"],
            "manifest_sha256": lock_record["manifest_sha256"],
            "splits_sha256": lock_record["splits_sha256"],
            "pretrained": {"overall": {"recall@5": 0.5, "mrr@10": 0.5}},
            "fine_tuned": {"overall": {"recall@5": 0.5, "mrr@10": 0.5}},
        }
        (out / "retrieval_evaluation.v1.json").write_text(json.dumps(report), encoding="utf-8")

        class FakeModel:
            def get_sentence_embedding_dimension(self):
                return 2

            def encode(self, texts, batch_size=None):
                return np.tile(np.array([[0.1, 0.2]]), (len(texts), 1))

        target = self.path / "new_index"
        with patch("src.finetuning.index.SentenceTransformer", return_value=FakeModel()):
            result = build_index(out, checkpoint, target, batch_size=1)
        self.assertEqual(result["chunks"], 2)
        client = chromadb.PersistentClient(path=str(target))
        collection = client.get_collection("policies_finetuned")
        self.assertEqual(collection.count(), 2)
        self.assertEqual(collection.metadata["manifest_sha256"], lock_record["manifest_sha256"])

    def test_synthetic_index_requires_explicit_opt_in_and_is_labeled(self):
        out = self.path / "synthetic_artifacts"
        out.mkdir()
        checkpoint = self.path / "synthetic_checkpoint"
        checkpoint.mkdir()
        (checkpoint / "model.safetensors").write_bytes(b"synthetic weights")
        chunks = [{
            "chunk_id": "c1", "policy_id": "p1", "title": "Policy one",
            "category": "Student Life", "type": "Regulation",
            "source_url": "https://policy.ufl.edu/policy/one/",
            "text": "Students must submit the form.",
        }]
        splits = [{"policy_id": "p1", "split": "test"}]
        write_jsonl(out / "manifest.v1.jsonl", chunks)
        write_jsonl(out / "policy_splits.v1.jsonl", splits)
        test = out / "assistant_reviewed_synthetic_test.v1.jsonl"
        write_jsonl(test, [{
            "question": "When must a student submit the form?",
            "draft_answer": "As stated in the passage.",
            "supporting_quote": "Students must submit the form.",
            "positive_chunk_id": "c1",
            "policy_id": "p1",
            "source_url": chunks[0]["source_url"],
            "question_type": "direct",
            "reviewed_by": "AI assistant",
            "review_note": "Synthetic trial question, not human reviewed.",
            "review_status": "approved",
            "synthetic": True,
        }])
        report = {
            "evaluation_type": "assistant_reviewed_synthetic_questions",
            "independent_human_test": False,
            "test_sha256": sha256_file(test),
            "manifest_sha256": sha256_file(out / "manifest.v1.jsonl"),
            "splits_sha256": sha256_file(out / "policy_splits.v1.jsonl"),
            "fine_tuned_model": str(checkpoint),
            "fine_tuned_fingerprint": model_fingerprint(checkpoint),
            "pretrained": {"overall": {"recall@5": 0.5, "mrr@10": 0.5}},
            "fine_tuned": {"overall": {"recall@5": 0.5, "mrr@10": 0.5}},
        }
        (out / "assistant_reviewed_synthetic_evaluation.v1.json").write_text(json.dumps(report))

        with self.assertRaisesRegex(ValueError, "locked human-test"):
            build_index(out, checkpoint, self.path / "human_index", batch_size=1)

        class FakeModel:
            def get_sentence_embedding_dimension(self):
                return 2

            def encode(self, texts, batch_size=None):
                return np.tile(np.array([[0.1, 0.2]]), (len(texts), 1))

        target = self.path / "synthetic_index"
        with patch("src.finetuning.index.SentenceTransformer", return_value=FakeModel()):
            result = build_index(out, checkpoint, target, batch_size=1, allow_synthetic_evaluation=True)
        client = chromadb.PersistentClient(path=str(target))
        collection = client.get_collection("policies_finetuned")
        self.assertEqual(result["chunks"], 1)
        self.assertEqual(collection.metadata["evaluation_type"], "assistant_reviewed_synthetic_questions")
        self.assertFalse(collection.metadata["independent_human_test"])
        self.assertEqual(
            collection.get(ids=["c1"], include=["metadatas"])["metadatas"][0]["type"],
            "Regulation",
        )

    def test_metrics(self):
        self.assertEqual(metric_at_rank(1)["recall@1"], 1)
        self.assertEqual(metric_at_rank(4)["recall@3"], 0)
        self.assertEqual(metric_at_rank(4)["recall@5"], 1)
        self.assertAlmostEqual(metric_at_rank(4)["mrr@10"], .25)
        self.assertAlmostEqual(metric_at_rank(4)["ndcg@10"], 1 / 2.321928094887362)
        self.assertEqual(metric_at_rank(11)["mrr@10"], 0)

    def test_rank_model_rejects_non_finite_embeddings(self):
        class FakeModel:
            def encode(self, texts, **kwargs):
                return np.full((len(texts), 2), np.inf, dtype=np.float32)

        with patch("src.finetuning.evaluate.SentenceTransformer", return_value=FakeModel()):
            with self.assertRaisesRegex(ValueError, "non-finite vectors"):
                rank_model("test-model", [{"chunk_id": "c1", "text": "source"}],
                           [{"question": "question", "positive_chunk_id": "c1"}], batch_size=1)

    def test_locked_test_rejects_wrong_split_and_changes(self):
        chunk = {"chunk_id": "c1", "policy_id": "p1", "category": "Student Life", "text": "Real text"}
        write_jsonl(self.path / "manifest.v1.jsonl", [chunk])
        write_jsonl(self.path / "policy_splits.v1.jsonl", [{"policy_id": "p1", "split": "test"}])
        test = self.path / "human_test.v1.jsonl"
        row = {"question": "What does this policy say?", "positive_chunk_id": "c1", "policy_id": "p1",
               "question_type": "direct", "reviewed_by": "human"}
        write_jsonl(test, [row])
        lock_test(self.path, test)
        self.assertEqual(len(validate_locked_test(self.path, test)), 1)
        row["question"] = "Changed question"
        write_jsonl(test, [row])
        with self.assertRaisesRegex(ValueError, "lock"):
            validate_locked_test(self.path, test)
        write_jsonl(test, [row])
        write_jsonl(self.path / "policy_splits.v1.jsonl", [{"policy_id": "p1", "split": "train"}])
        with self.assertRaisesRegex(ValueError, "lock"):
            validate_locked_test(self.path, test)

    def test_synthetic_test_is_labeled_and_checked_against_held_out_passage(self):
        chunk = {
            "chunk_id": "c1", "policy_id": "p1", "category": "Student Life",
            "source_url": "https://policy.ufl.edu/policy/test/", "text": "Students must submit the form.",
        }
        write_jsonl(self.path / "manifest.v1.jsonl", [chunk])
        write_jsonl(self.path / "policy_splits.v1.jsonl", [{"policy_id": "p1", "split": "test"}])
        test = self.path / "assistant_test.v1.jsonl"
        row = {
            "question": "When must a student submit the form?",
            "draft_answer": "As stated in the passage.",
            "supporting_quote": "Students must submit the form.",
            "positive_chunk_id": "c1",
            "policy_id": "p1",
            "source_url": chunk["source_url"],
            "question_type": "direct",
            "reviewed_by": "AI assistant",
            "review_note": "Source checked; not independently reviewed by a human.",
            "review_status": "approved",
            "synthetic": True,
        }
        write_jsonl(test, [row])
        metrics = {"overall": {"recall@5": 1.0, "mrr@10": 1.0}, "queries": []}
        with patch("src.finetuning.evaluate.rank_model", return_value=metrics):
            result = evaluate_synthetic_test(self.path, test, checkpoint=None, batch_size=8)
        report = json.loads(Path(result["report"]).read_text())
        self.assertEqual(report["evaluation_type"], "assistant_reviewed_synthetic_questions")
        self.assertFalse(report["independent_human_test"])
        self.assertEqual(report["test_questions"], 1)
        with self.assertRaisesRegex(ValueError, "human-authored"):
            lock_test(self.path, test)

        row["supporting_quote"] = "A quote absent from the source."
        write_jsonl(test, [row])
        with self.assertRaisesRegex(ValueError, "quote"):
            validate_synthetic_test(self.path, test)
        row["supporting_quote"] = chunk["text"]
        row["synthetic"] = False
        write_jsonl(test, [row])
        with self.assertRaisesRegex(ValueError, "explicitly synthetic"):
            validate_synthetic_test(self.path, test)


if __name__ == "__main__":
    unittest.main()
