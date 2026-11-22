# Local baselines

Model and prompt selection before formal training. Five base-model
configurations were evaluated with oMLX on the 631-question development set,
with temperature 1.0 and at most five turns per question. Each run is a single
pass and does not measure variance across repeated sampling.

| Run | Model | Thinking | Prompt | EM | F1 |
|---|---|---|---|---:|---:|
| [base-q35-original-n100](evaluations/base-q35-original-n100/) | Qwen3.5-4B | Off | original | 32.6% | 38.9% |
| [base-q35-fast-n100](evaluations/base-q35-fast-n100/) | Qwen3.5-4B | Off | fast | 33.3% | 40.8% |
| [base-q35-think-fast-n100](evaluations/base-q35-think-fast-n100/) | Qwen3.5-4B | On | fast | 37.2% | 44.9% |
| [base-q3i-original-n100](evaluations/base-q3i-original-n100/) | Qwen3-4B-Instruct-2507 | — | original | 23.9% | 32.0% |
| [base-q3i-fast-n100](evaluations/base-q3i-fast-n100/) | Qwen3-4B-Instruct-2507 | — | fast | 20.4% | 26.1% |

These results led to choosing Qwen3.5-4B for the main reproduction.

[Experiment metadata](experiment.json)
