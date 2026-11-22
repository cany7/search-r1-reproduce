"""Look inside one evaluation run: why trajectories ended, and what the ones that ran
out of turns were doing.

For every trajectory it checks whether a gold answer appears in the retrieved
documents (after the same normalization the scorer uses) and, if so, in which turn
it first appeared. That separates "had the evidence but kept searching" from
"never found it". Questions whose gold answer is already in the question (e.g.
"which of A and B ...") are left out of that check, because the answer string then
shows up in the documents whether or not the evidence does.

It also counts tool calls that were answered with an error (bad arguments), which
use up a turn without searching.

Usage: uv run python scripts/diagnose_run.py runs/eval/<run_name> [--examples 3]
"""

import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from search_r1.env.tools import normalize_answer


def gold_in_question(record: dict) -> bool:
    question = f" {normalize_answer(record['question'])} "
    return any(f" {normalize_answer(a)} " in question for a in record["gold_answers"] if a)


def is_tool_error(message: dict) -> bool:
    return message["role"] == "tool" and str(message["content"]).startswith('{"error"')


def first_turn_with_gold(record: dict) -> int | None:
    """1-based index of the first tool result whose documents contain a gold answer."""
    golds = [g for g in (normalize_answer(a) for a in record["gold_answers"]) if g]
    turn = 0
    for message in record["messages"]:
        if message["role"] != "tool":
            continue
        turn += 1
        # Drop the echoed "Query: ..." lines; only the documents count as evidence.
        lines = [ln for ln in str(message["content"]).split("\n") if not ln.startswith("Query: ")]
        text = f" {normalize_answer(' '.join(lines))} "
        if any(f" {gold} " in text for gold in golds):
            return turn
    return None


def share(count: int, total: int) -> str:
    return f"{count:>4} ({count / total:.0%})" if total else "   0"


def main() -> None:
    args = sys.argv[1:]
    num_examples = int(args[args.index("--examples") + 1]) if "--examples" in args else 3
    run_dir = Path(args[0])
    path = run_dir / "trajectories.jsonl"
    if not path.exists() and path.with_suffix(".jsonl.gz").exists():
        path = path.with_suffix(".jsonl.gz")
    with gzip.open(path, "rt") if path.suffix == ".gz" else open(path) as f:
        records = [json.loads(line) for line in f]
    records = [r for r in records if "error" not in r]
    n = len(records)
    for record in records:
        record["gold_in_question"] = gold_in_question(record)
        record["gold_turn"] = None if record["gold_in_question"] else first_turn_with_gold(record)
        record["tool_errors"] = sum(is_tool_error(m) for m in record["messages"])

    print(f"{run_dir.name}: {n} trajectories\n")
    print("Termination by data source (share of the source's trajectories)")
    terminations = [t for t, _ in Counter(r["termination"] for r in records).most_common()]
    by_source: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_source[record["data_source"]].append(record)
    print(f"{'':<17}" + "".join(f"{t[:15]:>17}" for t in terminations))
    for source, group in sorted(by_source.items()) + [("OVERALL", records)]:
        counts = Counter(r["termination"] for r in group)
        print(f"{source:<17}" + "".join(f"{counts[t] / len(group):>17.2f}" for t in terminations))

    print("\nWas a gold answer in the retrieved documents?")
    print("(checked = trajectories whose gold answer is not already in the question)")
    print(
        f"{'termination':<18}{'n':>5}{'checked':>9}{'gold retrieved':>17}{'tool error':>13}{'EM':>8}"
    )
    for termination in terminations:
        group = [r for r in records if r["termination"] == termination]
        checked = [r for r in group if not r["gold_in_question"]]
        found = sum(r["gold_turn"] is not None for r in checked)
        errors = sum(r["tool_errors"] > 0 for r in group)
        em = sum(r["em"] for r in group) / len(group)
        print(
            f"{termination:<18}{len(group):>5}{len(checked):>9}{share(found, len(checked)):>17}"
            f"{share(errors, len(group)):>13}{em:>8.3f}"
        )

    out_of_turns = [r for r in records if r["termination"] == "max_turns"]
    if out_of_turns:
        had_gold = [r for r in out_of_turns if r["gold_turn"] is not None]
        all_errors = [r for r in out_of_turns if r["tool_errors"] >= r["num_search_calls"] > 0]
        repeated = sum(r["repeated_query"] for r in out_of_turns)
        em_now = sum(r["em"] for r in records) / n
        print(f"\nRan out of turns: {share(len(out_of_turns), n)} of all trajectories")
        print(f"  gold answer already retrieved:   {share(len(had_gold), len(out_of_turns))}")
        print(
            "  turn in which it first appeared:",
            dict(sorted(Counter(r["gold_turn"] for r in had_gold).items())),
        )
        print(f"  every call was a tool error:     {share(len(all_errors), len(out_of_turns))}")
        print(f"  repeated a query:                {share(repeated, len(out_of_turns))}")
        print(
            "  If these had answered correctly once the gold answer was retrieved, EM would be "
            f"at most {em_now + len(had_gold) / n:.3f} (now {em_now:.3f})."
        )
        for record in had_gold[:num_examples]:
            print(f"\n  [{record['data_source']} {record['id']}] {record['question']}")
            print(
                f"    gold: {record['gold_answers'][:3]} | first retrieved in turn {record['gold_turn']}"
            )
            print(f"    queries: {record['queries']}")


if __name__ == "__main__":
    main()
