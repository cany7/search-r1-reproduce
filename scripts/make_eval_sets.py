"""Build the fixed development and final-test sets from the test split.

For each data source the questions are shuffled once with a fixed seed. The final
test set takes the first ``test_per_source`` of that order and the development set
the next ``dev_per_source``, so the two never overlap. A source with fewer questions
than ``test_per_source + dev_per_source`` is split in the same proportion.

Within each file, questions of a source keep the shuffled order, so the first k per
source (``offline_eval``'s ``limit_per_source``) is itself a fixed subset.

Usage: uv run python scripts/make_eval_sets.py [test_per_source=300] [dev_per_source=100]
"""

import json
import random
from collections import defaultdict
from pathlib import Path

import chz

from search_r1.env.search_env import SearchR1Datum, download_search_r1_dataset
from search_r1.env.tools import normalize_answer

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@chz.chz
class Config:
    test_per_source: int = 300
    dev_per_source: int = 100
    seed: int = 20261115
    output_dir: str = str(PROJECT_ROOT / "eval_sets")


def main(config: Config) -> None:
    data = download_search_r1_dataset("test")
    by_source: dict[str, list[SearchR1Datum]] = defaultdict(list)
    seen_questions: set[str] = set()
    num_duplicates = 0
    for item in data:
        # The same question can appear in more than one source; keep the first so
        # the development and test sets cannot share a question.
        key = normalize_answer(item["question"])
        if key in seen_questions:
            num_duplicates += 1
            continue
        seen_questions.add(key)
        by_source[item["data_source"]].append(item)
    print(f"{len(data)} questions, {num_duplicates} duplicates dropped")

    test_set: list[SearchR1Datum] = []
    dev_set: list[SearchR1Datum] = []
    for source in sorted(by_source):
        items = by_source[source]
        random.Random(f"{config.seed}-{source}").shuffle(items)
        wanted = config.test_per_source + config.dev_per_source
        if len(items) >= wanted:
            num_test, num_dev = config.test_per_source, config.dev_per_source
        else:
            num_test = round(len(items) * config.test_per_source / wanted)
            num_dev = len(items) - num_test
        test_set.extend(items[:num_test])
        dev_set.extend(items[num_test : num_test + num_dev])
        print(f"{source:<17} available {len(items):>6}  test {num_test:>4}  dev {num_dev:>4}")

    # ids restart from 0 in every source, so identify a question by (source, id).
    test_keys = {(i["data_source"], i["id"]) for i in test_set}
    dev_keys = {(i["data_source"], i["id"]) for i in dev_set}
    assert len(test_keys) == len(test_set) and len(dev_keys) == len(dev_set)
    assert not test_keys & dev_keys
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, items in (("test", test_set), ("dev", dev_set)):
        with open(output_dir / f"{name}.jsonl", "w") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"wrote {len(items)} items to {output_dir / f'{name}.jsonl'}")


if __name__ == "__main__":
    main(chz.entrypoint(Config))
