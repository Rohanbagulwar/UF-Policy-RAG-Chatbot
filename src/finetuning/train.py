"""Train an isolated question-to-passage BGE checkpoint from reviewed pairs."""
from __future__ import annotations

import argparse
import json
import math
import random
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

from .common import ARTIFACTS, BASE_MODEL, BASE_REVISION, read_jsonl, sha256_file
from .qa import normalize_question, validate_example


def training_pairs(out: Path) -> list[InputExample]:
    chunks = {row["chunk_id"]: row for row in read_jsonl(out / "manifest.v1.jsonl")}
    splits = {row["policy_id"]: row["split"] for row in read_jsonl(out / "policy_splits.v1.jsonl")}
    path = out / "qa_drafts.v1.jsonl"
    examples = read_jsonl(path)
    questions, pairs, used_passages = set(), [], set()
    for example in examples:
        validate_example(example, chunks, questions)
        questions.add(normalize_question(example["question"]))
        policy_split = splits.get(example["policy_id"])
        if policy_split not in ("train", "development"):
            raise ValueError("QA example does not belong to a train or development policy")
        if example.get("split") not in (None, policy_split):
            raise ValueError("QA example split does not match the policy split")
        if example["review_status"] != "approved" or policy_split == "development":
            continue
        if policy_split != "train":
            raise ValueError("Approved training example is not from a train policy")
        if example["positive_chunk_id"] in used_passages:
            continue  # one positive passage per batch avoids false in-batch negatives
        used_passages.add(example["positive_chunk_id"])
        pairs.append(InputExample(texts=[example["question"], chunks[example["positive_chunk_id"]]["text"]]))
    if not pairs:
        raise ValueError("No human-approved train pairs; review QA drafts first")
    return pairs


def train(out: Path, checkpoint: Path, device: str, epochs: int, batch_size: int, lr: float, seed: int, max_steps: int) -> dict:
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if batch_size < 2:
        raise ValueError("batch_size must be at least 2 to provide in-batch negatives")
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("learning rate must be positive")
    if max_steps < 0:
        raise ValueError("max_steps must be zero (uncapped) or positive")
    if device not in ("auto", "mps", "cuda", "cpu"):
        raise ValueError("device must be auto, mps, cuda, or cpu")
    if checkpoint.exists():
        raise FileExistsError(f"Checkpoint already exists: {checkpoint}")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    pairs = training_pairs(out)
    if len(pairs) < 2:
        raise ValueError("MultipleNegativesRankingLoss needs at least two distinct passages")
    model = SentenceTransformer(BASE_MODEL, revision=BASE_REVISION, device=device)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(pairs, shuffle=True, batch_size=batch_size, drop_last=True, generator=generator)
    if len(loader) == 0:
        raise ValueError("Batch size exceeds approved pair count")
    objective = losses.MultipleNegativesRankingLoss(model)
    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix=f".{checkpoint.name}.train-", dir=checkpoint.parent) as staging:
        model.fit(train_objectives=[(loader, objective)], epochs=epochs, optimizer_params={"lr": lr},
                  warmup_steps=0, steps_per_epoch=min(len(loader), max_steps) if max_steps else None,
                  output_path=staging, show_progress_bar=True)
        meta = {"base_model": BASE_MODEL, "base_revision": BASE_REVISION,
                "checkpoint": str(checkpoint), "embedding_dimension": model.get_sentence_embedding_dimension(),
                "loss": "MultipleNegativesRankingLoss", "device": device, "epochs": epochs, "batch_size": batch_size,
                "learning_rate": lr, "seed": seed, "max_steps_per_epoch": max_steps, "approved_pairs": len(pairs),
                "duration_seconds": round(time.monotonic() - start, 2),
                "manifest_sha256": sha256_file(out / "manifest.v1.jsonl"),
                "splits_sha256": sha256_file(out / "policy_splits.v1.jsonl"),
                "qa_sha256": sha256_file(out / "qa_drafts.v1.jsonl")}
        (Path(staging) / "training_metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        Path(staging).rename(checkpoint)
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ARTIFACTS)
    parser.add_argument("--checkpoint", type=Path, default=ARTIFACTS / "checkpoints/pilot_v1")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=10, help="Pilot cap; use 0 for full epoch")
    args = parser.parse_args()
    print(json.dumps(train(args.out, args.checkpoint, args.device, args.epochs, args.batch_size, args.learning_rate, args.seed, args.max_steps), indent=2))
