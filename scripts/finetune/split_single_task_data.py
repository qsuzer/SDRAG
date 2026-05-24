#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path


def classify_task(record):
    messages = record.get("messages", [])
    if len(messages) < 3:
        return "other"

    system = str(messages[0].get("content", "")).lower()
    user = str(messages[1].get("content", "")).lower()
    assistant = str(messages[2].get("content", "")).lower()

    if "breaking down complex questions into simpler sub-questions" in system:
        return "decomposition"
    if "analyzing logical dependencies between questions" in system or "json:" in assistant:
        if "sub-questions:" in user:
            return "dependency"
    if "sub-question to evaluate" in user:
        return "pruning"
    return "other"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input jsonl path")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--prefix", default="aligned_train_data", help="Output filename prefix")
    parser.add_argument("--report", help="Optional report json path")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "decomposition": output_dir / f"{args.prefix}_decomposition.jsonl",
        "dependency": output_dir / f"{args.prefix}_dependency.jsonl",
        "pruning": output_dir / f"{args.prefix}_pruning.jsonl",
    }

    counts = Counter()
    unknown_records = []

    handles = {task: path.open("w", encoding="utf-8") for task, path in outputs.items()}
    try:
        with input_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                record = json.loads(line)
                task = classify_task(record)
                counts[task] += 1
                if task in handles:
                    handles[task].write(json.dumps(record, ensure_ascii=False) + "\n")
                elif len(unknown_records) < 20:
                    unknown_records.append({"line_no": line_no, "task": task})
    finally:
        for handle in handles.values():
            handle.close()

    report = {
        "input": str(input_path),
        "outputs": {task: str(path) for task, path in outputs.items()},
        "counts": dict(counts),
        "unknown_records": unknown_records,
    }

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
