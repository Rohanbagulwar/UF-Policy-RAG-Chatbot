"""Resumable, human-reviewed synthetic QA drafts from real policy chunks."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import random
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from dotenv import load_dotenv

from .common import ARTIFACTS, ROOT, append_jsonl, read_jsonl, stable_id

load_dotenv(ROOT / ".env")

STYLES = ("direct", "procedure", "eligibility", "exception", "definition", "student wording")


def normalize_question(question: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", question.casefold()).strip()


def validate_example(example: dict, chunks: dict[str, dict], questions: set[str]) -> None:
    if not isinstance(example, dict):
        raise ValueError("Example must be a JSON object")
    required = {"question", "draft_answer", "supporting_quote", "positive_chunk_id", "policy_id",
                "title", "source_url", "original_source_url", "chunk_text", "question_type", "synthetic", "review_status"}
    if required - example.keys():
        raise ValueError(f"Missing fields: {sorted(required - example.keys())}")
    for key in ("question", "draft_answer", "supporting_quote", "positive_chunk_id", "policy_id",
                "title", "source_url", "original_source_url", "chunk_text", "question_type", "review_status"):
        if not isinstance(example[key], str):
            raise ValueError(f"{key} must be a string")
    chunk = chunks.get(example["positive_chunk_id"])
    if chunk is None:
        raise ValueError("Unknown chunk ID")
    for key in ("policy_id", "title", "source_url", "original_source_url"):
        if example[key] != chunk[key]:
            raise ValueError(f"Mismatched {key}")
    if example["chunk_text"] != chunk["text"]:
        raise ValueError("Mismatched source passage")
    quote = example["supporting_quote"].strip()
    if not quote or quote not in chunk["text"]:
        raise ValueError("Supporting quote is absent from passage")
    if not isinstance(example["draft_answer"], str) or not example["draft_answer"].strip():
        raise ValueError("Empty answer")
    if not isinstance(example["question"], str) or len(example["question"].strip()) < 8:
        raise ValueError("Malformed question")
    key = normalize_question(example["question"])
    if key in questions or any(SequenceMatcher(None, key, prior).ratio() >= .92 for prior in questions):
        raise ValueError("Duplicate or near-duplicate question")
    if example["question_type"] not in STYLES or example["synthetic"] is not True:
        raise ValueError("Invalid generation metadata")
    if example["review_status"] not in ("draft", "approved", "rejected"):
        raise ValueError("Invalid review status")
    if "split" in example and example["split"] not in ("train", "development"):
        raise ValueError("Synthetic examples must come from train or development policies")
    if example["review_status"] in ("approved", "rejected"):
        if not isinstance(example.get("reviewed_by"), str) or not example["reviewed_by"].strip():
            raise ValueError("Reviewed examples require a reviewer name")
        if not isinstance(example.get("review_note"), str) or not example["review_note"].strip():
            raise ValueError("Reviewed examples require a review note")


def select_pilot(chunks: list[dict], splits: dict[str, str], count: int, seed: int) -> list[dict]:
    """Choose train and development chunks from separate policies."""
    if count < 0:
        raise ValueError("count must not be negative")
    rng = random.Random(seed)

    def choose(split: str, limit: int) -> list[dict]:
        pool = [row for row in chunks if splits.get(row["policy_id"]) == split]
        rng.shuffle(pool)
        first, used = [], set()
        for row in pool:
            if row["policy_id"] not in used:
                first.append(row)
                used.add(row["policy_id"])
        return (first + [row for row in pool if row["policy_id"] in used and row not in first])[:limit]
    dev_count = min(round(count * .2), count)
    return choose("train", count - dev_count) + choose("development", dev_count)


class InvalidModelResponse(ValueError):
    pass


def _parse_response(response) -> dict:
    if not getattr(response, "choices", None):
        raise InvalidModelResponse("Gateway returned no completion choices")
    choice = response.choices[0]
    message = choice.message
    finish_reason = getattr(choice, "finish_reason", "unknown")
    raw = message.content
    if not isinstance(raw, str) or not raw.strip():
        usage = getattr(response, "usage", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        raise InvalidModelResponse(
            "Gateway returned empty message content "
            f"(finish_reason={finish_reason}, completion_tokens={completion_tokens}, "
            f"refusal={bool(getattr(message, 'refusal', None))})"
        )

    clean = raw.strip()
    if clean.startswith("```"):
        clean = re.sub(r"\A```(?:json)?\s*|\s*```\Z", "", clean, flags=re.IGNORECASE).strip()
    try:
        fields = json.loads(clean)
    except json.JSONDecodeError:
        start = clean.find("{")
        if start < 0:
            raise InvalidModelResponse(
                f"Gateway returned non-JSON content (finish_reason={finish_reason})"
            )
        try:
            fields, _ = json.JSONDecoder().raw_decode(clean[start:])
        except json.JSONDecodeError as error:
            usage = getattr(response, "usage", None)
            completion_tokens = getattr(usage, "completion_tokens", None)
            raise InvalidModelResponse(
                "Gateway returned incomplete/invalid JSON "
                f"(finish_reason={finish_reason}, completion_tokens={completion_tokens})"
            ) from error
    if not isinstance(fields, dict):
        raise InvalidModelResponse("Gateway JSON response was not an object")
    required = {"question", "draft_answer", "supporting_quote"}
    if required - fields.keys():
        raise InvalidModelResponse("Gateway JSON response is missing required fields")
    return fields


def _generate_one(
    client,
    chunk: dict,
    style: str,
    model: str,
    split: str,
    correction: bool = False,
) -> dict:
    from openai import BadRequestError

    correction_instruction = (
        " Your prior quote was not an exact match. Copy a short sentence verbatim from the passage, "
        "preserving punctuation and characters exactly."
        if correction else ""
    )
    prompt = ("Create ONE realistic student question and a short draft answer using ONLY the supplied UF policy passage. "
              "The question must be answerable by this passage alone. Return a JSON object with exactly "
              "question, draft_answer, supporting_quote. The quote must be an exact contiguous substring "
              "of the passage and must directly support the answer. Do not invent policy facts."
              f"{correction_instruction} Question style: {style}.\nTitle: {chunk['title']}\nPassage:\n{chunk['text']}")
    last_error = None
    fields = None
    for max_tokens in (1200, 2000):
        try:
            request = {
                "model": model,
                "temperature": 0.2,
                "max_tokens": max_tokens,
                "response_format": {"type": "json_object"},
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Return one compact JSON object only. Do not include analysis or markdown. "
                            "Generated answers are drafts, not official UF advice."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            }
            try:
                response = client.chat.completions.create(**request)
            except BadRequestError as error:
                message = str(error).casefold()
                if "response_format" not in message and "json" not in message:
                    raise
                request.pop("response_format")
                response = client.chat.completions.create(**request)
            fields = _parse_response(response)
            break
        except InvalidModelResponse as error:
            last_error = error
    if fields is None:
        raise InvalidModelResponse(str(last_error or "Gateway did not return valid JSON"))
    return {
        "question": fields["question"],
        "draft_answer": fields["draft_answer"],
        "supporting_quote": fields["supporting_quote"],
        "positive_chunk_id": chunk["chunk_id"],
        "policy_id": chunk["policy_id"],
        "title": chunk["title"],
        "source_url": chunk["source_url"],
        "original_source_url": chunk["original_source_url"],
        "chunk_text": chunk["text"],
        "question_type": style,
        "synthetic": True,
        "review_status": "draft",
        "generator_model": model,
        "split": split,
    }


def generate(
    out: Path,
    count: int = 250,
    seed: int = 42,
    model: str = "gpt-oss-20b",
    workers: int = 2,
) -> dict:
    """Generate drafts concurrently; append each valid result so runs can resume."""
    from openai import APIError, OpenAI

    if count < 0:
        raise ValueError("count must not be negative")
    if not 1 <= workers <= 8:
        raise ValueError("workers must be between 1 and 8")
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is required for QA generation")
    chunks = read_jsonl(out / "manifest.v1.jsonl")
    split_rows = read_jsonl(out / "policy_splits.v1.jsonl")
    splits = {row["policy_id"]: row["split"] for row in split_rows}
    path = out / "qa_drafts.v1.jsonl"
    drafts = read_jsonl(path) if path.exists() else []
    initial_draft_count = len(drafts)
    failures_path = out / "qa_failures.v1.jsonl"
    by_id = {row["chunk_id"]: row for row in chunks}
    questions = set()
    for item in drafts:
        validate_example(item, by_id, questions)
        questions.add(normalize_question(item["question"]))
    # The existing answer-comparison workbook has already been exposed during development.
    # Prevent its questions from entering the new training data.
    import pandas as pd
    old_eval = ROOT / "results/Accuracy_scores/Golden_dataset_question_answers_accuracy.xlsx"
    if old_eval.exists():
        old_questions = pd.read_excel(old_eval)["Question"].dropna()
        questions.update(normalize_question(str(value)) for value in old_questions)
    completed = {item["positive_chunk_id"] for item in drafts}
    client = OpenAI(
        api_key=key,
        base_url=os.environ.get("UF_AI_BASE_URL", "https://api.ai.it.ufl.edu"),
        max_retries=2,
        timeout=45.0,
    )
    failures = Counter()
    selected = select_pilot(chunks, splits, count, seed)
    pending = [
        (index, chunk)
        for index, chunk in enumerate(selected)
        if chunk["chunk_id"] not in completed
    ]
    executor = ThreadPoolExecutor(max_workers=workers)
    futures = {}
    try:
        futures = {
            executor.submit(
                _generate_one,
                client,
                chunk,
                STYLES[index % len(STYLES)],
                model,
                splits[chunk["policy_id"]],
            ): (chunk, index)
            for index, chunk in pending
        }
        for finished, future in enumerate(as_completed(futures), start=1):
            chunk, index = futures[future]
            try:
                item = future.result()
                try:
                    validate_example(item, by_id, questions)
                except ValueError as validation_error:
                    if "Supporting quote is absent from passage" not in str(validation_error):
                        raise
                    print(f"Retrying chunk {chunk['chunk_id']}: generated quote did not exactly match.")
                    item = _generate_one(
                        client,
                        chunk,
                        STYLES[index % len(STYLES)],
                        model,
                        splits[chunk["policy_id"]],
                        correction=True,
                    )
                    validate_example(item, by_id, questions)
                item["example_id"] = stable_id(chunk["chunk_id"], item["question"])
                drafts.append(item)
                questions.add(normalize_question(item["question"]))
                completed.add(chunk["chunk_id"])
                append_jsonl(path, item)
            except (APIError, json.JSONDecodeError, KeyError, ValueError, TypeError, IndexError, AttributeError) as error:
                message = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}\b", "sk-[REDACTED]", str(error))[:500]
                reason = f"{type(error).__name__}: {message}"
                failures[reason] += 1
                append_jsonl(failures_path, {
                    "chunk_id": chunk["chunk_id"],
                    "error_type": type(error).__name__,
                    "error": message,
                })
                print(f"Failed chunk {chunk['chunk_id']}: {reason}")
            print(
                f"Completed {finished}/{len(pending)} requests; "
                f"saved {len(drafts)} drafts; {sum(failures.values())} failed."
            )
    except KeyboardInterrupt:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        print(f"Interrupted; saved {len(drafts)} drafts. Rerun the same command to resume.")
        raise
    else:
        executor.shutdown(wait=True)
    return {
        "drafts": len(drafts),
        "new_drafts": len(drafts) - initial_draft_count,
        "failures": dict(failures),
        "failure_log": str(failures_path),
        "path": str(path),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ARTIFACTS)
    parser.add_argument("--count", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="gpt-oss-20b")
    parser.add_argument("--workers", type=int, default=2,
                        help="Concurrent API requests (1-8; increase cautiously if rate limits allow)")
    args = parser.parse_args()
    try:
        result = generate(args.out, args.count, args.seed, args.model, args.workers)
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    print(json.dumps(result, indent=2))
