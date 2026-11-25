"""Build a provenance-preserving manifest from the existing workbook."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import pandas as pd

from .common import ARTIFACTS, ROOT, WORKBOOK, sha256_file, stable_id, write_jsonl


def policy_url(value: object) -> str:
    raw = str(value).strip()
    parts = urlsplit(raw)
    if parts.scheme != "https" or parts.hostname != "policy.ufl.edu" or parts.username or parts.password or parts.port or not parts.path.startswith(("/policy/", "/regulation/")):
        raise ValueError(f"Invalid official policy URL: {raw}")
    return urlunsplit(("https", "policy.ufl.edu", parts.path.rstrip("/") + "/", "", ""))


def parse_workbook(path: Path) -> tuple[list[dict], dict]:
    frame = pd.read_excel(path, sheet_name="Policies")
    required = {"title", "category", "type", "link", "chunk_number", "content"}
    if required - set(frame.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(frame.columns))}")
    rows, rejected, seen_keys, seen_text = [], Counter(), set(), set()
    for _, item in frame.iterrows():
        try:
            url = policy_url(item["link"])
            text = str(item["content"]) if pd.notna(item["content"]) else ""
            title = str(item["title"]).strip() if pd.notna(item["title"]) else ""
            category = str(item["category"]).strip() if pd.notna(item["category"]) else ""
            document_type = str(item["type"]).strip() if pd.notna(item["type"]) else ""
            if not text.strip() or text.casefold() == "nan":
                raise ValueError("empty text")
            if not title or not category or not document_type:
                raise ValueError("missing title, category, or type")
            number = int(item["chunk_number"])
            if number < 1 or float(item["chunk_number"]) != number:
                raise ValueError("bad chunk number")
            key = (url, number)
            content_key = (url, text.casefold())
            if key in seen_keys or content_key in seen_text:
                raise ValueError("duplicate chunk")
            seen_keys.add(key)
            seen_text.add(content_key)
            rows.append({"chunk_id": stable_id(url, str(number), text), "policy_id": stable_id(url), "source_url": url,
                         "original_source_url": str(item["link"]).strip(), "title": title,
                         "category": category, "type": document_type,
                         "chunk_number": number, "text": text})
        except (ValueError, TypeError) as error:
            rejected[str(error)] += 1
    rows.sort(key=lambda row: (row["source_url"], row["chunk_number"]))
    try:
        source_workbook = path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        source_workbook = path.name
    meta = {"schema_version": 1, "source_workbook": source_workbook, "source_sha256": sha256_file(path),
            "chunker": "RecursiveCharacterTextSplitter; chunk_size=1000; chunk_overlap=100 (notebook settings)",
            "accepted_chunks": len(rows), "policies": len({row["policy_id"] for row in rows}), "rejected": dict(rejected)}
    return rows, meta


def split_policies(rows: list[dict], seed: int = 42) -> dict[str, str]:
    """Stratify by primary category while assigning each URL to one split."""
    by_policy = {}
    for row in rows:
        by_policy.setdefault(row["policy_id"], row["category"])
    groups = defaultdict(list)
    for policy_id, category in by_policy.items():
        groups[category].append(policy_id)
    rng = random.Random(seed)
    result = {}
    for category, ids in sorted(groups.items()):
        rng.shuffle(ids)
        n = len(ids)
        n_test = round(n * .15) if n >= 3 else 0
        n_dev = round(n * .15) if n >= 3 else 0
        for i, policy_id in enumerate(ids):
            result[policy_id] = "test" if i < n_test else "development" if i < n_test + n_dev else "train"
    if len(result) < 3:
        raise ValueError("At least three distinct policies are required for train/development/test splits")
    for required_split in ("test", "development"):
        if required_split not in result.values():
            candidates = sorted(policy for policy, split in result.items() if split == "train")
            if not candidates:
                raise ValueError(f"Unable to create a {required_split} policy split")
            result[candidates[-1]] = required_split
    return result


def build(path: Path = WORKBOOK, out: Path = ARTIFACTS, seed: int = 42) -> dict:
    rows, meta = parse_workbook(path)
    splits = split_policies(rows, seed)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "manifest.v1.jsonl", rows)
    split_rows = [{"policy_id": policy, "split": split} for policy, split in sorted(splits.items())]
    write_jsonl(out / "policy_splits.v1.jsonl", split_rows)
    meta.update({"split_seed": seed, "split_counts": dict(Counter(splits.values())),
                 "manifest_sha256": sha256_file(out / "manifest.v1.jsonl"),
                 "splits_sha256": sha256_file(out / "policy_splits.v1.jsonl")})
    (out / "manifest_metadata.v1.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook", type=Path, default=WORKBOOK)
    parser.add_argument("--out", type=Path, default=ARTIFACTS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(build(args.workbook, args.out, args.seed), indent=2))
