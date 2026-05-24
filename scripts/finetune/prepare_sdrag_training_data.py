"""Standalone Experiment 3 data preparation script.

This version is intentionally lightweight:

1. Read the mixed `aligned_train_data.jsonl`.
2. Use natural pruning KEEP samples as seeds.
3. Generate synthetic pruning negatives with three methods:
   - intent_shift
   - entity_replace
   - over_decompose
4. Merge them back into the mixed training file.
5. Materialize a truly balanced 1:1:1 mixed training file.

It does NOT run the archive's expensive teacher re-labeling or judge filtering.
Only lightweight cleanup is kept:
- dedup against existing pruning sub-questions
- dedup within generated negatives
- max synthetic samples per question

The script is standalone on purpose and does not import from `flashrag`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import re
import string
import uuid
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from openai import AsyncOpenAI


PUNCT_TRANS = str.maketrans("", "", string.punctuation)

TASK_DECOMP = "decomposition"
TASK_DEP = "dependency"
TASK_PRUNE = "pruning"

DECOMP_SYSTEM_MARKER = "breaking down complex questions into simpler sub-questions"
DEP_SYSTEM_MARKER = "analyzing logical dependencies between questions"
PRUNE_SYSTEM_MARKER = "evaluating reasoning chains"
PRUNE_USER_MARKER = "sub-question to evaluate:"


INTENT_SHIFT_PROMPT = """
You are generating training data for a sub-question pruning model.

Given an original multi-hop question and one of its USEFUL sub-questions,
generate a NEW sub-question that:
1. Keeps the SAME main subject / entity as the seed sub-question.
2. Asks about a DIFFERENT attribute, relation, or property.
3. Is NOT needed to answer the original question.
4. Sounds natural, grammatically correct, and topically related.
5. Should be plausible enough that a careless reader might think it's useful.

Return a JSON object with exactly these keys:
{
  "sub_question": "...",
  "reason": "one short sentence explaining why it is not needed"
}

Original question:
{question}

Useful sub-question (seed):
{seed_sub_question}
""".strip()


ENTITY_REPLACE_PROMPT = """
You are generating training data for a sub-question pruning model.

Given an original multi-hop question and one of its USEFUL sub-questions,
generate a NEW sub-question that:
1. Keeps the same question structure / template as the seed.
2. Replaces the KEY entity with a DIFFERENT entity of the same type.
3. The replacement entity must NOT be relevant to answering the original question.
4. Prefer entities from the same domain but clearly different.
5. The result must be grammatically correct.

Return a JSON object with exactly these keys:
{
  "sub_question": "...",
  "reason": "one short sentence explaining why it is not needed"
}

Original question:
{question}

Useful sub-question (seed):
{seed_sub_question}
""".strip()


OVER_DECOMPOSE_PROMPT = """
You are generating training data for a sub-question pruning model.

Given an original multi-hop question, produce an intentionally over-decomposed
reasoning plan:
1. Break it into AT LEAST {min_sub_questions} sub-questions.
2. Include both essential steps and extra supplementary steps.
3. Make every sub-question natural and grammatically correct.

Return a JSON object with exactly this schema:
{
  "sub_questions": ["...", "...", "..."]
}

Original question:
{question}
""".strip()


@dataclass
class PruningSeed:
    question: str
    sub_question: str
    metadata: dict[str, Any]


@dataclass
class Candidate:
    question: str
    candidate_sub_question: str
    source_method: str
    metadata: dict[str, Any]
    seed_sub_question: str = ""
    reason: str = ""


class AsyncChatClient:
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
        max_parallel: int = 8,
        temperature: float = 0.7,
        max_tokens: int = 256,
        retries: int = 5,
        retry_backoff: float = 2.0,
    ) -> None:
        self.model = model
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.semaphore = asyncio.Semaphore(max_parallel)
        self.max_parallel = max_parallel
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retries = retries
        self.retry_backoff = retry_backoff

    async def generate(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        for attempt in range(self.retries):
            try:
                async with self.semaphore:
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        response_format={"type": "json_object"},
                    )
                content = response.choices[0].message.content or ""
                return content.strip()
            except Exception:
                if attempt == self.retries - 1:
                    raise
                await asyncio.sleep(self.retry_backoff * (attempt + 1))
        raise RuntimeError("unreachable")

    async def batch(self, prompts: list[str], stage_name: str = "generation") -> list[Any]:
        if not prompts:
            return []
        results: list[Any] = []
        total = len(prompts)
        for start in range(0, total, self.max_parallel):
            end = min(start + self.max_parallel, total)
            chunk = prompts[start:end]
            chunk_tasks = [self.generate(prompt) for prompt in chunk]
            chunk_results = await asyncio.gather(*chunk_tasks, return_exceptions=True)
            results.extend(chunk_results)
            errors = sum(1 for item in chunk_results if isinstance(item, BaseException))
            log(f"{stage_name}: {end}/{total} completed (chunk_errors={errors})")
        return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone Experiment 3 data prep: pruning negatives + 1:1:1 resampling."
    )
    parser.add_argument("--input-path", required=True, help="Path to aligned_train_data.jsonl")
    parser.add_argument("--output-dir", required=True, help="Directory for generated outputs")
    parser.add_argument("--model", default="gpt-4o", help="Model used for negative construction")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", ""))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-parallel", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--intent-weight", type=float, default=0.4)
    parser.add_argument("--entity-weight", type=float, default=0.3)
    parser.add_argument("--over-weight", type=float, default=0.3)
    parser.add_argument("--max-per-question", type=int, default=2)
    parser.add_argument("--max-original-subqs", type=int, default=4)
    parser.add_argument("--min-over-sub-questions", type=int, default=6)
    parser.add_argument("--balance-target", default="min", help='"min" or integer count per task')
    parser.add_argument("--target-synthetic-count", type=int, default=0, help="0 means use all available seeds")
    parser.add_argument("--candidate-overgen-factor", type=float, default=1.5)
    parser.add_argument("--disable-intent-shift", action="store_true")
    parser.add_argument("--disable-entity-replace", action="store_true")
    parser.add_argument("--disable-over-decompose", action="store_true")
    return parser.parse_args()


def render_prompt(template: str, **values: str) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace(f"{{{key}}}", value)
    return rendered.strip()


def normalize(text: str) -> str:
    return text.lower().translate(PUNCT_TRANS).strip()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    print(f"[{utc_now()}] {message}", flush=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_progress(path: Path, **fields: Any) -> None:
    payload = {}
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
    payload.update(fields)
    payload["updated_at"] = utc_now()
    write_json(path, payload)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def classify_task(record: dict[str, Any]) -> str:
    messages = record.get("messages", [])
    if len(messages) < 2:
        raise ValueError("Record does not contain enough messages")
    system = str(messages[0].get("content", "")).lower()
    user = str(messages[1].get("content", "")).lower()
    assistant = str(messages[-1].get("content", "")).lower()

    if PRUNE_SYSTEM_MARKER in system or PRUNE_USER_MARKER in user:
        return TASK_PRUNE
    if DEP_SYSTEM_MARKER in system or ("sub-questions:" in user and "json" in assistant):
        return TASK_DEP
    if DECOMP_SYSTEM_MARKER in system:
        return TASK_DECOMP
    raise ValueError(f"Unable to classify task from metadata: {record.get('metadata', {})}")


def extract_pruning_question_and_subquestion(record: dict[str, Any]) -> tuple[str, str]:
    user = str(record["messages"][1]["content"])
    question_match = re.search(r"Original question:\s*(.+?)(?:\n|$)", user, re.IGNORECASE | re.DOTALL)
    subq_match = re.search(r"Sub-question to evaluate:\s*(.+)$", user, re.IGNORECASE | re.DOTALL)
    if not question_match or not subq_match:
        raise ValueError(f"Unable to parse pruning user message: {user[:200]}")
    return question_match.group(1).strip(), subq_match.group(1).strip()


def extract_pruning_label(record: dict[str, Any]) -> str:
    assistant = str(record["messages"][2]["content"])
    match = re.search(r"Decision:\s*(KEEP|PRUNE)", assistant, re.IGNORECASE)
    if not match:
        raise ValueError(f"Unable to parse pruning label: {assistant[:200]}")
    return match.group(1).upper()


def extract_decomposition_question(record: dict[str, Any]) -> str:
    user = str(record["messages"][1]["content"])
    match = re.search(r"Complex Question:\s*(.+)$", user, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    lines = [line.strip() for line in user.splitlines() if line.strip()]
    return lines[-1]


def extract_decomposition_subquestions(record: dict[str, Any]) -> list[str]:
    assistant = str(record["messages"][2]["content"])
    subqs: list[str] = []
    in_section = False
    for raw_line in assistant.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        normalized = line.lower().lstrip("#").strip()
        if normalized.startswith("sub-questions"):
            in_section = True
            continue
        if not in_section:
            continue
        match = re.match(r"^\d+[\).\s-]+(.+)$", line)
        if match:
            subqs.append(match.group(1).strip())
            continue
        bullet_match = re.match(r"^[-*]\s+(.+)$", line)
        if bullet_match:
            subqs.append(bullet_match.group(1).strip())
    if not subqs:
        raise ValueError(f"Unable to parse decomposition sub-questions: {assistant[:200]}")
    return subqs


def extract_first_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found")
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : index + 1])
    raise ValueError("Unclosed JSON object")


def build_index(
    records: list[dict[str, Any]]
) -> tuple[list[PruningSeed], dict[str, dict[str, Any]], str, Counter]:
    pruning_seeds: list[PruningSeed] = []
    decomposition_by_question: dict[str, dict[str, Any]] = {}
    pruning_system_prompt = ""
    counts: Counter = Counter()

    for record in records:
        task = classify_task(record)
        counts[task] += 1

        if task == TASK_PRUNE:
            if not pruning_system_prompt:
                pruning_system_prompt = str(record["messages"][0]["content"])
            label = extract_pruning_label(record)
            if label != "KEEP":
                continue
            question, sub_question = extract_pruning_question_and_subquestion(record)
            pruning_seeds.append(
                PruningSeed(
                    question=question,
                    sub_question=sub_question,
                    metadata=deepcopy(record.get("metadata", {})),
                )
            )
        elif task == TASK_DECOMP:
            question = extract_decomposition_question(record)
            decomposition_by_question[question] = {
                "sub_questions": extract_decomposition_subquestions(record),
                "metadata": deepcopy(record.get("metadata", {})),
            }

    if not pruning_system_prompt:
        raise ValueError("No pruning system prompt found in input dataset")
    return pruning_seeds, decomposition_by_question, pruning_system_prompt, counts


def existing_pruning_subquestions(records: list[dict[str, Any]]) -> set[str]:
    subqs: set[str] = set()
    for record in records:
        if classify_task(record) != TASK_PRUNE:
            continue
        _, sub_question = extract_pruning_question_and_subquestion(record)
        subqs.add(sub_question)
    return subqs


def select_overdecompose_questions(
    decomposition_by_question: dict[str, dict[str, Any]],
    max_original_subqs: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for question, payload in decomposition_by_question.items():
        sub_questions = payload["sub_questions"]
        if len(sub_questions) <= max_original_subqs:
            selected.append(
                {
                    "question": question,
                    "original_sub_questions": sub_questions,
                    "metadata": deepcopy(payload.get("metadata", {})),
                }
            )
    return selected


def safe_json_loads(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return extract_first_json_object(text)


async def generate_candidates(
    seeds: list[PruningSeed],
    over_questions: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[Candidate]:
    client = AsyncChatClient(
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url or None,
        max_parallel=args.max_parallel,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    rng = random.Random(args.seed)
    shuffled_seeds = list(seeds)
    rng.shuffle(shuffled_seeds)
    shuffled_over_questions = list(over_questions)
    rng.shuffle(shuffled_over_questions)

    active_weights = []
    if not args.disable_intent_shift:
        active_weights.append(("intent_shift", args.intent_weight))
    if not args.disable_entity_replace:
        active_weights.append(("entity_replace", args.entity_weight))
    if not args.disable_over_decompose:
        active_weights.append(("over_decompose", args.over_weight))

    total_weight = sum(weight for _, weight in active_weights)
    assignments: dict[str, list[PruningSeed]] = {"intent_shift": [], "entity_replace": []}
    selected_over_questions: list[dict[str, Any]] = []

    if args.target_synthetic_count > 0 and total_weight > 0:
        prompt_budget = max(1, math.ceil(args.target_synthetic_count * args.candidate_overgen_factor))
        budgets = {
            name: max(0, int(prompt_budget * weight / total_weight))
            for name, weight in active_weights
        }
        allocated = sum(budgets.values())
        if allocated < prompt_budget:
            first_name = active_weights[0][0]
            budgets[first_name] += prompt_budget - allocated
    else:
        budgets = {name: None for name, _ in active_weights}

    seed_only = [(name, weight) for name, weight in active_weights if name in {"intent_shift", "entity_replace"}]
    total_seed_weight = sum(weight for _, weight in seed_only)
    cursor = 0
    for index, (name, weight) in enumerate(seed_only):
        budget = budgets.get(name)
        if index == len(seed_only) - 1:
            selected = shuffled_seeds[cursor:]
        else:
            portion = int(len(shuffled_seeds) * weight / max(total_seed_weight, 1e-9))
            selected = shuffled_seeds[cursor : cursor + portion]
            cursor += portion
        if budget is not None:
            selected = selected[:budget]
        assignments[name] = selected

    if not args.disable_over_decompose and shuffled_over_questions:
        over_budget = budgets.get("over_decompose")
        if over_budget is None:
            selected_over_questions = shuffled_over_questions
        else:
            selected_over_questions = shuffled_over_questions[:over_budget]

    prompts: list[str] = []
    prompt_meta: list[tuple[str, Any]] = []

    for seed in assignments.get("intent_shift", []):
        prompts.append(render_prompt(INTENT_SHIFT_PROMPT, question=seed.question, seed_sub_question=seed.sub_question))
        prompt_meta.append(("intent_shift", seed))
    for seed in assignments.get("entity_replace", []):
        prompts.append(render_prompt(ENTITY_REPLACE_PROMPT, question=seed.question, seed_sub_question=seed.sub_question))
        prompt_meta.append(("entity_replace", seed))
    for payload in selected_over_questions:
        prompts.append(
            render_prompt(
                OVER_DECOMPOSE_PROMPT,
                question=payload["question"],
                min_sub_questions=str(args.min_over_sub_questions),
            )
        )
        prompt_meta.append(("over_decompose", payload))

    if not prompts:
        return []

    method_prompt_counts = Counter(name for name, _ in prompt_meta)
    log(f"starting candidate generation with prompt_counts={dict(method_prompt_counts)}")
    responses = await client.batch(prompts, stage_name="candidate_generation")
    candidates: list[Candidate] = []

    for meta, response in zip(prompt_meta, responses):
        if isinstance(response, BaseException):
            continue
        method, payload = meta
        try:
            obj = safe_json_loads(str(response))
        except Exception:
            continue

        if method in {"intent_shift", "entity_replace"}:
            seed = payload
            sub_question = str(obj.get("sub_question", "")).strip()
            reason = str(obj.get("reason", "")).strip()
            if not sub_question:
                continue
            candidates.append(
                Candidate(
                    question=seed.question,
                    candidate_sub_question=sub_question,
                    source_method=method,
                    metadata=deepcopy(seed.metadata),
                    seed_sub_question=seed.sub_question,
                    reason=reason,
                )
            )
            continue

        original_norms = {normalize(subq) for subq in payload["original_sub_questions"]}
        for sub_question in obj.get("sub_questions", []):
            if not isinstance(sub_question, str):
                continue
            if normalize(sub_question) in original_norms:
                continue
            candidates.append(
                Candidate(
                    question=payload["question"],
                    candidate_sub_question=sub_question.strip(),
                    source_method=method,
                    metadata=deepcopy(payload.get("metadata", {})),
                    reason="This is an extra over-decomposed step that is not required to answer the original question.",
                )
            )

    log(f"candidate generation finished with {len(candidates)} raw candidates")
    return candidates


def dedup_and_cap(
    candidates: list[Candidate],
    existing_sub_qs: set[str],
    max_per_question: int,
) -> list[Candidate]:
    seen = {normalize(subq) for subq in existing_sub_qs}
    per_question: Counter = Counter()
    results: list[Candidate] = []

    for candidate in candidates:
        normalized = normalize(candidate.candidate_sub_question)
        if not normalized:
            continue
        if normalized in seen:
            continue
        if candidate.seed_sub_question and normalized == normalize(candidate.seed_sub_question):
            continue
        if per_question[candidate.question] >= max_per_question:
            continue

        seen.add(normalized)
        per_question[candidate.question] += 1
        results.append(candidate)

    return results


def build_prune_thought(candidate: Candidate) -> str:
    if candidate.reason:
        return candidate.reason
    if candidate.source_method == "intent_shift":
        return "This candidate keeps the same entity but shifts to a different attribute that is not required for the original question."
    if candidate.source_method == "entity_replace":
        return "This candidate preserves the question pattern but replaces a key entity with a different one that is not relevant to the original question."
    return "This candidate is an extra over-decomposed step and is not required to answer the original question."


def prune_user_message(question: str, sub_question: str) -> str:
    return f"Original question: {question}\n     Sub-question to evaluate: {sub_question}"


def prune_assistant_message(candidate: Candidate) -> str:
    return f"Thought: {build_prune_thought(candidate)}\nDecision: PRUNE"


def candidate_to_record(candidate: Candidate, pruning_system_prompt: str) -> dict[str, Any]:
    metadata = {
        "source": candidate.metadata.get("source", "synthetic"),
        "type": candidate.metadata.get("type", "synthetic"),
        "synthetic": True,
        "source_method": candidate.source_method,
        "synthetic_id": str(uuid.uuid4()),
    }
    return {
        "messages": [
            {"role": "system", "content": pruning_system_prompt},
            {"role": "user", "content": prune_user_message(candidate.question, candidate.candidate_sub_question)},
            {"role": "assistant", "content": prune_assistant_message(candidate)},
        ],
        "metadata": metadata,
    }


def resolve_target_count(counts: Counter, balance_target: str) -> int:
    if balance_target == "min":
        return min(counts.values())
    return int(balance_target)


def build_balanced_records(records: list[dict[str, Any]], balance_target: str, seed: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_task[classify_task(record)].append(record)

    counts = Counter({task: len(items) for task, items in by_task.items()})
    if set(counts) != {TASK_DECOMP, TASK_DEP, TASK_PRUNE}:
        raise ValueError(f"Expected three tasks, got {dict(counts)}")

    target = resolve_target_count(counts, balance_target)
    rng = random.Random(seed)
    balanced: list[dict[str, Any]] = []
    effective_counts: dict[str, int] = {}

    for task in (TASK_DECOMP, TASK_DEP, TASK_PRUNE):
        items = list(by_task[task])
        rng.shuffle(items)
        if len(items) < target:
            repeats = math.ceil(target / max(len(items), 1))
            items = (items * repeats)[:target]
        else:
            items = items[:target]
        effective_counts[task] = len(items)
        balanced.extend(items)

    rng.shuffle(balanced)
    return balanced, effective_counts


async def prepare_dataset(args: argparse.Namespace) -> dict[str, Any]:
    if not args.api_key:
        raise ValueError("Missing API key. Pass --api-key or set OPENAI_API_KEY.")

    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.json"
    write_progress(progress_path, stage="starting", input_path=str(input_path))

    log(f"reading input from {input_path}")
    records = read_jsonl(input_path)
    log(f"loaded {len(records)} total records")
    write_progress(progress_path, stage="loaded_records", total_records=len(records))

    seeds, decomposition_by_question, pruning_system_prompt, original_counts = build_index(records)
    over_questions = select_overdecompose_questions(decomposition_by_question, args.max_original_subqs)
    existing_sub_qs = existing_pruning_subquestions(records)
    natural_keep = len(seeds)
    natural_prune = original_counts[TASK_PRUNE] - natural_keep

    log(
        "indexed records: "
        f"counts={dict(original_counts)}, keep_seeds={natural_keep}, over_questions={len(over_questions)}"
    )
    write_progress(
        progress_path,
        stage="indexed",
        original_counts=dict(original_counts),
        natural_pruning_keep_count=natural_keep,
        natural_pruning_prune_count=natural_prune,
        seed_count=len(seeds),
        over_decompose_question_count=len(over_questions),
    )

    generated_candidates = await generate_candidates(seeds, over_questions, args)
    write_progress(
        progress_path,
        stage="generated_candidates",
        generated_candidate_count=len(generated_candidates),
    )

    final_candidates = dedup_and_cap(generated_candidates, existing_sub_qs, args.max_per_question)
    log(f"after dedup/cap: {len(final_candidates)} candidates remain")
    if args.target_synthetic_count > 0 and len(final_candidates) > args.target_synthetic_count:
        rng = random.Random(args.seed)
        rng.shuffle(final_candidates)
        final_candidates = final_candidates[: args.target_synthetic_count]
        log(f"trimmed final candidates to target_synthetic_count={args.target_synthetic_count}")

    write_progress(
        progress_path,
        stage="finalized_candidates",
        final_synthetic_count=len(final_candidates),
    )

    synthetic_records = [candidate_to_record(candidate, pruning_system_prompt) for candidate in final_candidates]
    augmented_records = list(records) + synthetic_records
    balanced_records, balanced_counts = build_balanced_records(augmented_records, args.balance_target, args.seed)

    method_counts = Counter(candidate.source_method for candidate in final_candidates)
    augmented_counts = Counter(classify_task(record) for record in augmented_records)
    log(
        "writing outputs: "
        f"synthetic={len(synthetic_records)}, augmented_counts={dict(augmented_counts)}, "
        f"balanced_counts={balanced_counts}"
    )

    report = {
        "input_path": str(input_path),
        "original_counts": dict(original_counts),
        "natural_pruning_keep_count": natural_keep,
        "natural_pruning_prune_count": natural_prune,
        "seed_count": len(seeds),
        "over_decompose_question_count": len(over_questions),
        "target_synthetic_count": args.target_synthetic_count,
        "generated_candidate_count": len(generated_candidates),
        "final_synthetic_count": len(final_candidates),
        "synthetic_by_method": dict(method_counts),
        "augmented_counts": dict(augmented_counts),
        "balanced_counts": balanced_counts,
    }

    synthetic_path = output_dir / "synthetic_pruning_records.jsonl"
    augmented_path = output_dir / "aligned_train_data_augmented.jsonl"
    balanced_path = output_dir / "aligned_train_data_balanced_1to1to1.jsonl"
    details_path = output_dir / "synthetic_pruning_details.jsonl"
    report_path = output_dir / "prepare_report.json"

    write_jsonl(synthetic_path, synthetic_records)
    write_jsonl(augmented_path, augmented_records)
    write_jsonl(balanced_path, balanced_records)

    detail_records = []
    for candidate in final_candidates:
        detail_records.append(
            {
                "question": candidate.question,
                "seed_sub_question": candidate.seed_sub_question,
                "candidate_sub_question": candidate.candidate_sub_question,
                "source_method": candidate.source_method,
                "reason": build_prune_thought(candidate),
                "metadata": candidate.metadata,
            }
        )
    write_jsonl(details_path, detail_records)
    write_json(report_path, report)
    write_progress(progress_path, stage="completed", report=report)
    log("data preparation completed successfully")
    return report


def main() -> None:
    args = parse_args()
    report = asyncio.run(prepare_dataset(args))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
