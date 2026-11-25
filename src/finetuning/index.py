"""Build a distinct Chroma index from real chunks, with model compatibility checks."""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from .common import ARTIFACTS, ROOT, read_jsonl, sha256_file
from .evaluate import LOCK_FILE, validate_synthetic_test


def model_fingerprint(checkpoint: Path) -> str:
    files = sorted(checkpoint.rglob("*.safetensors"))
    if not files:
        raise ValueError("Checkpoint has no safetensors weights")
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(checkpoint)).encode())
        digest.update(sha256_file(path).encode())
    return digest.hexdigest()


def compatible(metadata: dict, checkpoint: Path, dimension: int, corpus_hash: str) -> None:
    expected = {"model_fingerprint": model_fingerprint(checkpoint), "embedding_dimension": dimension,
                "manifest_sha256": corpus_hash}
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Index compatibility failure: {key}")


def guarded_query(collection, checkpoint: Path, question: str, corpus_hash: str, n_results: int = 5) -> dict:
    return FineTunedRetriever(collection, checkpoint, corpus_hash).query(question, n_results)


class FineTunedRetriever:
    """Load and validate a fine-tuned model once for repeated collection queries."""

    def __init__(self, collection, checkpoint: Path, corpus_hash: str):
        self.collection = collection
        self.model = SentenceTransformer(str(checkpoint))
        self.dimension = self.model.get_sentence_embedding_dimension()
        compatible(collection.metadata or {}, checkpoint, self.dimension, corpus_hash)

    def query(self, question: str, n_results: int = 5) -> dict:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        if n_results < 1:
            raise ValueError("n_results must be at least 1")
        vector = self.model.encode([question])[0]
        if len(vector) != self.dimension:
            raise ValueError("Query vector dimension mismatch")
        return self.collection.query(query_embeddings=[vector.tolist()], n_results=n_results)


def build(
    out: Path,
    checkpoint: Path,
    persist: Path,
    collection_name: str = "policies_finetuned",
    batch_size: int = 32,
    allow_synthetic_evaluation: bool = False,
) -> dict:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    baseline = (ROOT / "data/chroma_db").resolve()
    target = persist.resolve()
    if target == baseline or baseline in target.parents or target in baseline.parents:
        raise ValueError("Fine-tuned persistence must be separate from baseline")
    if collection_name == "policies":
        raise ValueError("Cannot use baseline collection name")
    if target.exists():
        raise FileExistsError("Target index directory already exists; refusing to mutate it")

    if allow_synthetic_evaluation:
        report_path = out / "assistant_reviewed_synthetic_evaluation.v1.json"
        test = out / "assistant_reviewed_synthetic_test.v1.jsonl"
        if not report_path.is_file():
            raise ValueError("Synthetic evaluation report is missing")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            report.get("evaluation_type") != "assistant_reviewed_synthetic_questions"
            or report.get("independent_human_test") is not False
        ):
            raise ValueError("Synthetic index requires an explicitly labeled synthetic evaluation")
        validate_synthetic_test(out, test)
        if (
            report.get("test_sha256") != sha256_file(test)
            or report.get("manifest_sha256") != sha256_file(out / "manifest.v1.jsonl")
            or report.get("splits_sha256") != sha256_file(out / "policy_splits.v1.jsonl")
        ):
            raise ValueError("Synthetic evaluation data has changed since evaluation")
    else:
        report_path = out / "retrieval_evaluation.v1.json"
        if not report_path.is_file():
            raise ValueError("A locked human-test retrieval evaluation is required")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        test = out / "human_test.v1.jsonl"
        lock = out / LOCK_FILE
        lock_record = json.loads(lock.read_text(encoding="utf-8")) if lock.exists() else {}
        test_hash = sha256_file(test) if test.exists() else None
        manifest_hash = sha256_file(out / "manifest.v1.jsonl")
        splits_hash = sha256_file(out / "policy_splits.v1.jsonl")
        if (
            report.get("test_sha256") != lock_record.get("test_sha256")
            or report.get("test_sha256") != test_hash
            or report.get("manifest_sha256") != lock_record.get("manifest_sha256")
            or report.get("manifest_sha256") != manifest_hash
        ):
            raise ValueError("The evaluated locked test has changed")
        if report.get("splits_sha256") != lock_record.get("splits_sha256") or report.get("splits_sha256") != splits_hash:
            raise ValueError("The evaluated policy splits have changed")

    if report.get("fine_tuned_model") != str(checkpoint):
        raise ValueError("Checkpoint has no matching retrieval evaluation")
    if report.get("fine_tuned_fingerprint") != model_fingerprint(checkpoint):
        raise ValueError("Checkpoint weights changed after retrieval evaluation")
    manifest_hash = sha256_file(out / "manifest.v1.jsonl")
    base = report["pretrained"]["overall"]
    tuned = report["fine_tuned"]["overall"]
    if tuned["recall@5"] < base["recall@5"] or tuned["mrr@10"] < base["mrr@10"]:
        raise ValueError("Fine-tuned model did not pass retrieval comparison")
    chunks = read_jsonl(out / "manifest.v1.jsonl")
    corpus_hash = manifest_hash
    if corpus_hash != report["manifest_sha256"]:
        raise ValueError("Evaluated corpus differs from index corpus")
    model = SentenceTransformer(str(checkpoint))
    dimension = model.get_sentence_embedding_dimension()
    metadata = {"model_identity": str(checkpoint), "model_fingerprint": model_fingerprint(checkpoint),
                "embedding_dimension": dimension, "manifest_sha256": corpus_hash,
                "splits_sha256": report["splits_sha256"], "test_sha256": report["test_sha256"],
                "evaluation_type": report.get("evaluation_type", "independent_human_test"),
                "independent_human_test": not allow_synthetic_evaluation,
                "chunker": "RecursiveCharacterTextSplitter 1000/100; workbook snapshot", "hnsw:space": "cosine"}
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{target.name}.build-", dir=target.parent) as staging:
        client = chromadb.PersistentClient(path=staging)
        collection = client.create_collection(name=collection_name, metadata=metadata)
        for offset in range(0, len(chunks), batch_size):
            batch = chunks[offset:offset + batch_size]
            vectors = model.encode([row["text"] for row in batch], batch_size=batch_size)
            collection.add(ids=[row["chunk_id"] for row in batch], documents=[row["text"] for row in batch],
                           embeddings=vectors.tolist(), metadatas=[{"policy_id": row["policy_id"], "title": row["title"],
                           "category": row["category"], "type": row.get("type", "Policy"),
                           "source_url": row["source_url"]} for row in batch])
        if collection.count() != len(chunks):
            raise RuntimeError("Fine-tuned index row count does not match the manifest")
        indexed_count = collection.count()
        compatible(collection.metadata or {}, checkpoint, dimension, corpus_hash)
        del collection, client
        Path(staging).rename(target)
    return {"path": str(target), "collection": collection_name, "chunks": indexed_count, "metadata": metadata}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ARTIFACTS)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--persist", type=Path)
    parser.add_argument("--collection")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--allow-synthetic-evaluation", action="store_true",
                        help="Build a clearly labeled trial index from synthetic evaluation; not for production")
    args = parser.parse_args()
    if args.allow_synthetic_evaluation:
        persist = args.persist or args.out / "chroma_finetuned_synthetic_v1"
        collection_name = args.collection or "policies_finetuned_synthetic"
    else:
        persist = args.persist or args.out / "chroma_finetuned_v1"
        collection_name = args.collection or "policies_finetuned"
    print(json.dumps(build(args.out, args.checkpoint, persist, collection_name, args.batch_size,
                           allow_synthetic_evaluation=args.allow_synthetic_evaluation), indent=2))
