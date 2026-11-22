# Search-R1 main reproduction

The main reproduction has two training stages: one under the original search
rule and one under the limited-search rule. The stage-2 step-75 checkpoint was
selected based on development-set results; the training record extended to
step 100 is also kept.

Limited-search rule: each turn executes only the first 3 valid queries in the
first search call.

| Training stage | Start → end | Search rule |
|---|---|---|
| [Stage 1](training/stage-1/) | Base → 50 | Original |
| [Stage 2](training/stage-2/) | 50 → 75 | Limited search |
| [Extended training](training/extended-training/) | 75 → 100 | Limited search |

Stage 2 and the extended training share the same
[training configuration](training/stage-2/config.json); only `max_steps`
differs (75 / 100).

The development set (dev) has 631 questions and the test set (test) has 1,894
questions. Evaluation records are organized by search rule, dataset and
checkpoint.

| Evaluation rule | Dataset | Checkpoints |
|---|---|---|
| Original | dev | [Base](evaluations/original-search/dev/base/), [step 25](evaluations/original-search/dev/step-0025/), [step 50](evaluations/original-search/dev/step-0050/) |
| Original | test | [Base](evaluations/original-search/test/base/), [step 25](evaluations/original-search/test/step-0025/), [step 50](evaluations/original-search/test/step-0050/) |
| Limited search | dev | [Step 25](evaluations/limited-search/dev/step-0025/), [step 50](evaluations/limited-search/dev/step-0050/), [step 75](evaluations/limited-search/dev/step-0075/), [step 100](evaluations/limited-search/dev/step-0100/) |
| Limited search | test | [Step 75](evaluations/limited-search/test/step-0075/), [step 100](evaluations/limited-search/test/step-0100/) |

The step-25 and step-50 evaluations under the limited-search rule use stage-1
checkpoints.

Per-question [trajectories](evaluations/original-search/test/step-0050/trajectories.jsonl.gz)
for the original-search step-50 test evaluation are provided gzip-compressed.

[Experiment metadata](experiment.json)
