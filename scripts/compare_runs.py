"""Print the summaries of several evaluation runs side by side.

Usage: uv run python scripts/compare_runs.py runs/eval/base-*
       uv run python scripts/compare_runs.py --by-source runs/eval/base-*
"""

import json
import sys
from pathlib import Path

TERMINATIONS = [
    "answered",
    "no_valid_answer",
    "max_turns",
    "turn_truncated",
    "context_overflow",
    "parse_error",
]


def fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def row(name: str, s: dict) -> str:
    termination = s.get("termination", {})
    shares = "".join(f"{termination.get(t, 0) / s['n']:>9.2f}" for t in TERMINATIONS)
    return (
        f"{name:<34}{s['n']:>5}{fmt(s['em']):>7}{fmt(s['f1']):>7}{fmt(s['answer_rate']):>7}"
        f"{fmt(s['em_when_answered']):>8}{s['search_calls']:>7.2f}{s['sampled_tokens']:>7.0f}{shares}"
    )


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--by-source"]
    by_source = "--by-source" in sys.argv
    header = (
        f"{'run':<34}{'N':>5}{'EM':>7}{'F1':>7}{'ans':>7}{'EM|ans':>8}{'srch':>7}{'tok':>7}"
        + "".join(f"{t[:8]:>9}" for t in TERMINATIONS)
    )
    print(header)
    print("-" * len(header))
    for path in args:
        summary_file = Path(path) / "summary.json"
        if not summary_file.exists():
            print(f"{Path(path).name:<34} (no summary.json: run not finished)")
            continue
        summary = json.loads(summary_file.read_text())
        print(row(Path(path).name, summary["by_source"]["OVERALL"]))
        if by_source:
            for source, s in summary["by_source"].items():
                if source != "OVERALL":
                    print(row(f"  {source}", s))
    print("\nans = share with a well-formed answer; srch = search calls; tok = sampled tokens;")
    print("the last columns are the share of each termination reason.")


if __name__ == "__main__":
    main()
