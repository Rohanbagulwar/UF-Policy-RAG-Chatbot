"""Frozen retrieval comparison over the same held-out queries and real passages."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from .common import ARTIFACTS, BASE_MODEL, BASE_REVISION, ROOT, read_jsonl, sha256_file
from .qa import normalize_question, validate_example


LOCK_FILE = "locked_test.v1.json"


def metric_at_rank(rank: int | None) -> dict[str, float]:
    return {"recall@1": float(rank is not None and rank <= 1),
            "recall@3": float(rank is not None and rank <= 3),
            "recall@5": float(rank is not None and rank <= 5),
            "mrr@10": 1 / rank if rank is not None and rank <= 10 else 0.0,
            "ndcg@10": 1 / np.log2(rank + 1) if rank is not None and rank <= 10 else 0.0}


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {}
    return {name: round(float(np.mean([row["metrics"][name] for row in rows])), 4)
            for name in metric_at_rank(None)}


def validate_locked_test(out: Path, test: Path) -> list[dict]:
    lock = out / LOCK_FILE
    if not lock.exists():
        raise ValueError("Test set is not locked; review and lock the independent test first")
    lock_record = json.loads(lock.read_text(encoding="utf-8"))
    expected_lock = {
        "schema_version": 1,
        "test_sha256": sha256_file(test),
        "manifest_sha256": sha256_file(out / "manifest.v1.jsonl"),
        "splits_sha256": sha256_file(out / "policy_splits.v1.jsonl"),
    }
    if lock_record != expected_lock:
        raise ValueError("Test set missing or changed since lock; review and lock the independent test first")
    chunks = {row["chunk_id"]: row for row in read_jsonl(out / "manifest.v1.jsonl")}
    splits = {row["policy_id"]: row["split"] for row in read_jsonl(out / "policy_splits.v1.jsonl")}
    questions = read_jsonl(test)
    if not questions:
        raise ValueError("Empty locked test")
    _validate_human_test_rows(questions, chunks, splits)
    old_eval = ROOT / "results/Accuracy_scores/Golden_dataset_question_answers_accuracy.xlsx"
    if old_eval.exists():
        import pandas as pd
        exposed = {normalize_question(str(value)) for value in pd.read_excel(old_eval)["Question"].dropna()}
        if any(normalize_question(row["question"]) in exposed for row in questions):
            raise ValueError("Locked test contains a previously exposed evaluation question")
    return questions


def validate_synthetic_test(out: Path, test: Path) -> list[dict]:
    chunks = {row["chunk_id"]: row for row in read_jsonl(out / "manifest.v1.jsonl")}
    splits = {row["policy_id"]: row["split"] for row in read_jsonl(out / "policy_splits.v1.jsonl")}
    questions = read_jsonl(test)
    if not questions:
        raise ValueError("Empty synthetic test")
    seen_questions = set()
    for row in questions:
        for field in ("question", "draft_answer", "supporting_quote", "positive_chunk_id",
                      "policy_id", "source_url", "question_type", "reviewed_by", "review_note"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"Synthetic test requires a non-empty string field: {field}")
        if row.get("synthetic") is not True or row.get("review_status") != "approved":
            raise ValueError("Synthetic test requires approved, explicitly synthetic questions")
        key = normalize_question(row["question"])
        if key in seen_questions:
            raise ValueError("Duplicate question in synthetic test")
        seen_questions.add(key)
        chunk = chunks.get(row["positive_chunk_id"])
        if (
            chunk is None
            or row["policy_id"] != chunk["policy_id"]
            or row["source_url"] != chunk["source_url"]
            or splits.get(chunk["policy_id"]) != "test"
            or row["supporting_quote"] not in chunk["text"]
        ):
            raise ValueError("Synthetic test label or quote does not match a held-out source passage")

    old_eval = ROOT / "results/Accuracy_scores/Golden_dataset_question_answers_accuracy.xlsx"
    if old_eval.exists():
        import pandas as pd
        exposed = {normalize_question(str(value)) for value in pd.read_excel(old_eval)["Question"].dropna()}
        if any(normalize_question(row["question"]) in exposed for row in questions):
            raise ValueError("Synthetic test contains a previously exposed evaluation question")
    return questions


def _validate_human_test_rows(questions: list[dict], chunks: dict[str, dict], splits: dict[str, str]) -> None:
    seen_questions = set()
    for row in questions:
        for field in ("question", "positive_chunk_id", "policy_id", "question_type", "reviewed_by"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"Test requires a non-empty string field: {field}")
        if row.get("synthetic") is True:
            raise ValueError("Independent test questions must be human-authored")
        question_key = " ".join(row["question"].casefold().split())
        if question_key in seen_questions:
            raise ValueError("Duplicate question in independent test")
        seen_questions.add(question_key)
        chunk = chunks.get(row.get("positive_chunk_id"))
        if (
            chunk is None
            or row.get("policy_id") != chunk["policy_id"]
            or splits.get(chunk["policy_id"]) != "test"
        ):
            raise ValueError("Test label points outside held-out test policies")


def lock_test(out: Path, test: Path) -> None:
    # Validate labels before locking; this is a deliberate, one-time human action.
    chunks = {row["chunk_id"]: row for row in read_jsonl(out / "manifest.v1.jsonl")}
    splits = {row["policy_id"]: row["split"] for row in read_jsonl(out / "policy_splits.v1.jsonl")}
    rows = read_jsonl(test)
    if not rows:
        raise ValueError("Cannot lock empty test")
    _validate_human_test_rows(rows, chunks, splits)
    # Exclude old development questions before freezing this test file.
    old_eval = ROOT / "results/Accuracy_scores/Golden_dataset_question_answers_accuracy.xlsx"
    if old_eval.exists():
        import pandas as pd
        exposed = {normalize_question(str(value)) for value in pd.read_excel(old_eval)["Question"].dropna()}
        if any(normalize_question(row["question"]) in exposed for row in rows):
            raise ValueError("Test contains a previously exposed evaluation question")
    lock = out / LOCK_FILE
    if lock.exists():
        raise FileExistsError("Test already locked")
    lock.write_text(json.dumps({
        "schema_version": 1,
        "test_sha256": sha256_file(test),
        "manifest_sha256": sha256_file(out / "manifest.v1.jsonl"),
        "splits_sha256": sha256_file(out / "policy_splits.v1.jsonl"),
    }, indent=2) + "\n", encoding="utf-8")


def rank_model(model_path: str, chunks: list[dict], questions: list[dict], batch_size: int) -> dict:
    model = SentenceTransformer(model_path, revision=BASE_REVISION if model_path == BASE_MODEL else None)
    passages = model.encode([row["text"] for row in chunks], batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)
    queries = model.encode([row["question"] for row in questions], batch_size=batch_size, normalize_embeddings=True)
    query_vectors = np.asarray(queries, dtype=np.float64)
    passage_vectors = np.asarray(passages, dtype=np.float64)
    if not np.isfinite(query_vectors).all() or not np.isfinite(passage_vectors).all():
        raise ValueError("Embedding model produced non-finite vectors")
    scores = np.einsum("qd,pd->qp", query_vectors, passage_vectors, optimize=False)
    if not np.isfinite(scores).all():
        raise ValueError("Embedding similarity calculation produced non-finite scores")
    ids = [row["chunk_id"] for row in chunks]
    chunks_by_id = {row["chunk_id"]: row for row in chunks}
    results = []
    for question, vector in zip(questions, scores):
        order = np.argsort(-vector, kind="stable")[:10]
        ranked = [ids[i] for i in order]
        rank = ranked.index(question["positive_chunk_id"]) + 1 if question["positive_chunk_id"] in ranked else None
        results.append({"question": question["question"], "positive_chunk_id": question["positive_chunk_id"],
                        "rank": rank, "top10_chunk_ids": ranked, "metrics": metric_at_rank(rank),
                        "category": chunks_by_id[question["positive_chunk_id"]]["category"],
                        "question_type": question.get("question_type", "unspecified")})
    groups = defaultdict(list)
    for row in results:
        groups[("category", row["category"])].append(row)
        groups[("question_type", row["question_type"])].append(row)
    return {"overall": summarize(results), "groups": {f"{key}:{value}": {"n": len(group), **summarize(group)} for (key, value), group in groups.items()}, "queries": results}


def evaluate(out: Path, test: Path, checkpoint: Path | None, batch_size: int) -> dict:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    questions = validate_locked_test(out, test)
    chunks = read_jsonl(out / "manifest.v1.jsonl")
    splits_hash = sha256_file(out / "policy_splits.v1.jsonl")
    report = {"test_sha256": sha256_file(test), "manifest_sha256": sha256_file(out / "manifest.v1.jsonl"),
              "splits_sha256": splits_hash,
              "candidate_chunks": len(chunks), "test_questions": len(questions),
              "base_model": BASE_MODEL, "base_revision": BASE_REVISION,
              "pretrained": rank_model(BASE_MODEL, chunks, questions, batch_size)}
    if checkpoint:
        from .index import model_fingerprint
        meta_path = checkpoint / "training_metadata.json"
        metadata = json.loads(meta_path.read_text())
        if metadata["manifest_sha256"] != report["manifest_sha256"]:
            raise ValueError("Checkpoint trained against another corpus")
        if metadata["splits_sha256"] != splits_hash:
            raise ValueError("Checkpoint trained with a different policy split")
        qa_path = out / "qa_drafts.v1.jsonl"
        if metadata["qa_sha256"] != sha256_file(qa_path):
            raise ValueError("Training QA data changed after checkpoint creation")
        if metadata["base_model"] != BASE_MODEL or metadata["base_revision"] != BASE_REVISION:
            raise ValueError("Checkpoint was not trained from the configured baseline model revision")
        report["fine_tuned_model"] = str(checkpoint)
        report["fine_tuned_fingerprint"] = model_fingerprint(checkpoint)
        report["fine_tuned"] = rank_model(str(checkpoint), chunks, questions, batch_size)
    destination = out / "retrieval_evaluation.v1.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    return {"report": str(destination), "pretrained": report["pretrained"]["overall"],
            "fine_tuned": report.get("fine_tuned", {}).get("overall")}


def evaluate_synthetic_test(out: Path, test: Path, checkpoint: Path | None, batch_size: int) -> dict:
    """Compare models on assistant-reviewed synthetic questions without calling them human test data."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    questions = validate_synthetic_test(out, test)
    chunks = read_jsonl(out / "manifest.v1.jsonl")
    splits_hash = sha256_file(out / "policy_splits.v1.jsonl")
    manifest_hash = sha256_file(out / "manifest.v1.jsonl")
    report = {
        "evaluation_type": "assistant_reviewed_synthetic_questions",
        "independent_human_test": False,
        "test_sha256": sha256_file(test),
        "manifest_sha256": manifest_hash,
        "splits_sha256": splits_hash,
        "candidate_chunks": len(chunks),
        "test_questions": len(questions),
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "pretrained": rank_model(BASE_MODEL, chunks, questions, batch_size),
    }
    if checkpoint:
        from .index import model_fingerprint
        meta_path = checkpoint / "training_metadata.json"
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if metadata["manifest_sha256"] != manifest_hash:
            raise ValueError("Checkpoint trained against another corpus")
        if metadata["splits_sha256"] != splits_hash:
            raise ValueError("Checkpoint trained with a different policy split")
        if metadata["qa_sha256"] != sha256_file(out / "qa_drafts.v1.jsonl"):
            raise ValueError("Training QA data changed after checkpoint creation")
        if metadata["base_model"] != BASE_MODEL or metadata["base_revision"] != BASE_REVISION:
            raise ValueError("Checkpoint was not trained from the configured baseline model revision")
        report["fine_tuned_model"] = str(checkpoint)
        report["fine_tuned_fingerprint"] = model_fingerprint(checkpoint)
        report["fine_tuned"] = rank_model(str(checkpoint), chunks, questions, batch_size)
    destination = out / "assistant_reviewed_synthetic_evaluation.v1.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return {"report": str(destination), "pretrained": report["pretrained"]["overall"],
            "fine_tuned": report.get("fine_tuned", {}).get("overall"),
            "evaluation_type": report["evaluation_type"]}


def evaluate_development(out: Path, checkpoint: Path | None, batch_size: int) -> dict:
    """Evaluate only human-approved synthetic questions assigned to development policies."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    chunks = read_jsonl(out / "manifest.v1.jsonl")
    chunk_by_id = {row["chunk_id"]: row for row in chunks}
    splits = {row["policy_id"]: row["split"] for row in read_jsonl(out / "policy_splits.v1.jsonl")}
    examples = read_jsonl(out / "qa_drafts.v1.jsonl")
    seen_questions = set()
    questions = []
    for example in examples:
        validate_example(example, chunk_by_id, seen_questions)
        seen_questions.add(normalize_question(example["question"]))
        policy_split = splits.get(example["policy_id"])
        if policy_split not in ("train", "development"):
            raise ValueError("Synthetic QA must not reference test or unknown policies")
        if example.get("split") not in (None, policy_split):
            raise ValueError("QA example split does not match the policy split")
        if example["review_status"] == "approved" and policy_split == "development":
            questions.append({**example, "reviewed_by": example["reviewed_by"]})
    if not questions:
        raise ValueError("No human-approved development questions are available")

    manifest_hash = sha256_file(out / "manifest.v1.jsonl")
    splits_hash = sha256_file(out / "policy_splits.v1.jsonl")
    qa_hash = sha256_file(out / "qa_drafts.v1.jsonl")
    report = {
        "evaluation_type": "development_only_synthetic_questions",
        "manifest_sha256": manifest_hash,
        "splits_sha256": splits_hash,
        "qa_sha256": qa_hash,
        "candidate_chunks": len(chunks),
        "development_questions": len(questions),
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "pretrained": rank_model(BASE_MODEL, chunks, questions, batch_size),
    }
    if checkpoint:
        from .index import model_fingerprint
        metadata = json.loads((checkpoint / "training_metadata.json").read_text(encoding="utf-8"))
        if metadata["manifest_sha256"] != manifest_hash or metadata["splits_sha256"] != splits_hash:
            raise ValueError("Checkpoint was trained against a different corpus or policy split")
        if metadata["qa_sha256"] != qa_hash:
            raise ValueError("Training QA data changed after checkpoint creation")
        if metadata["base_model"] != BASE_MODEL or metadata["base_revision"] != BASE_REVISION:
            raise ValueError("Checkpoint was not trained from the configured baseline model revision")
        report["fine_tuned_model"] = str(checkpoint)
        report["fine_tuned_fingerprint"] = model_fingerprint(checkpoint)
        report["fine_tuned"] = rank_model(str(checkpoint), chunks, questions, batch_size)
    destination = out / "development_evaluation.v1.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return {"report": str(destination), "pretrained": report["pretrained"]["overall"],
            "fine_tuned": report.get("fine_tuned", {}).get("overall")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ARTIFACTS)
    parser.add_argument("--test", type=Path, default=ARTIFACTS / "human_test.v1.jsonl")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lock-test", action="store_true")
    parser.add_argument("--synthetic-test", action="store_true",
                        help="Evaluate explicitly synthetic, assistant-reviewed questions (not an independent human test)")
    parser.add_argument("--development", action="store_true",
                        help="Evaluate human-approved development questions instead of the locked test")
    args = parser.parse_args()
    if sum((args.lock_test, args.synthetic_test, args.development)) > 1:
        parser.error("--lock-test, --synthetic-test, and --development cannot be used together")
    if args.development:
        print(json.dumps(evaluate_development(args.out, args.checkpoint, args.batch_size), indent=2))
    elif args.synthetic_test:
        print(json.dumps(evaluate_synthetic_test(args.out, args.test, args.checkpoint, args.batch_size), indent=2))
    elif args.lock_test:
        lock_test(args.out, args.test)
        print("Locked", args.test)
    else:
        print(json.dumps(evaluate(args.out, args.test, args.checkpoint, args.batch_size), indent=2))
