#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def parse_history(run_dir: Path):
    logging_path = run_dir / "logging.jsonl"
    if logging_path.is_file():
        return load_jsonl(logging_path)

    histories = []
    for ckpt_dir in sorted(run_dir.glob("checkpoint-*")):
        state_path = ckpt_dir / "trainer_state.json"
        if not state_path.is_file():
            continue
        state = load_json(state_path)
        for entry in state.get("log_history", []):
            if isinstance(entry, dict):
                histories.append(entry)
    return histories


def maybe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def maybe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def resolve_step(entry):
    step = maybe_int(entry.get("step") or entry.get("global_step"))
    if step is not None:
        return step

    step_text = entry.get("global_step/max_steps")
    if isinstance(step_text, str):
        match = re.match(r"\s*(\d+)\s*/\s*\d+\s*", step_text)
        if match:
            return int(match.group(1))
    return None


def resolve_best_checkpoint(run_dir: Path, step: int | None):
    if step is not None:
        candidate = run_dir / f"checkpoint-{step}"
        if candidate.is_dir():
            return candidate

    checkpoints = sorted(
        [
            path
            for path in run_dir.glob("checkpoint-*")
            if path.is_dir() and path.name.split("-")[-1].isdigit()
        ],
        key=lambda path: int(path.name.split("-")[-1]),
    )
    if not checkpoints:
        return None
    if step is None:
        return checkpoints[-1]

    best = min(
        checkpoints,
        key=lambda path: abs(int(path.name.split("-")[-1]) - step),
    )
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument(
        "--format",
        choices=("tsv", "json"),
        default="tsv",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    history = parse_history(run_dir)

    eval_entries = []
    train_entries = []
    seen_eval = set()

    for entry in history:
        if not isinstance(entry, dict):
            continue

        if "eval_loss" in entry:
            eval_loss = maybe_float(entry.get("eval_loss"))
            step = resolve_step(entry)
            key = (step, eval_loss)
            if key not in seen_eval:
                seen_eval.add(key)
                eval_entries.append(
                    {
                        "step": step,
                        "eval_loss": eval_loss,
                        "eval_token_acc": maybe_float(entry.get("eval_token_acc")),
                    }
                )

        if "loss" in entry:
            train_entries.append(
                {
                    "step": resolve_step(entry),
                    "loss": maybe_float(entry.get("loss")),
                }
            )

    last_eval = eval_entries[-1]["eval_loss"] if eval_entries else None
    best_eval_entry = min(
        (entry for entry in eval_entries if entry["eval_loss"] is not None),
        key=lambda entry: entry["eval_loss"],
        default=None,
    )
    last_train = next(
        (entry["loss"] for entry in reversed(train_entries) if entry["loss"] is not None),
        None,
    )

    best_step = best_eval_entry["step"] if best_eval_entry else None
    best_ckpt = resolve_best_checkpoint(run_dir, best_step)

    result = {
        "run_dir": str(run_dir),
        "last_eval_loss": last_eval,
        "best_eval_loss": best_eval_entry["eval_loss"] if best_eval_entry else None,
        "best_eval_step": best_step,
        "best_eval_token_acc": best_eval_entry["eval_token_acc"] if best_eval_entry else None,
        "best_checkpoint": str(best_ckpt) if best_ckpt else "",
        "last_train_loss": last_train,
        "num_eval_points": len(eval_entries),
    }

    if args.format == "json":
        print(json.dumps(result, ensure_ascii=False))
        return

    values = [
        result["last_eval_loss"],
        result["best_eval_loss"],
        result["best_eval_step"],
        result["best_eval_token_acc"],
        result["best_checkpoint"],
        result["last_train_loss"],
        result["num_eval_points"],
    ]
    print("\t".join("" if value is None else str(value) for value in values))


if __name__ == "__main__":
    main()
