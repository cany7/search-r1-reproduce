# Local base-model evaluation analysis

[中文版](BASELINE_ANALYSIS_zh.md)

Analysis date: 2026-11-16. This report uses the saved configurations, summaries and
trajectories from five `base-*-n100` runs. Archiving did not rerun the models or
change the original scores or prompts. Results are under `evaluations/` in this
directory; their original location was `runs/eval/`.

These are local model-selection results collected before formal training. The
Qwen3.5 non-thinking runs used `qwen3_5_disable_thinking`, the thinking run used
`qwen3_5`, and Qwen3-Instruct used `qwen3_instruct`. All predate the Unicode renderer
correction in `1b26fd7`; they do not measure the new renderer. The original records
do not reliably identify the project's execution-time Git commit. The archive
commit must not be substituted for that missing information.

## Data integrity and comparison scope

- Each run contains the same 631 `(data_source, id)` pairs: 100 questions from each
  of six sources and 31 from Bamboogle. Every run reports `num_errors=0`. This means
  no request exceptions; parsing failures are separately recorded terminations.
- Shared settings: temperature 1.0, `max_turns=5`, `max_tokens=1024`,
  `max_trajectory_tokens=32768`, concurrency 8, and the same BM25 service. No context
  overflow was recorded. Both local models were BF16 variants; exact names are in
  the run configurations.
- These are single oMLX runs on the development set, used for model selection and
  behavioral diagnosis. They are not held-out test results or repeated-sampling
  estimates of stability.
- EM uses strict answer extraction. Format compliance requires exactly one
  case-sensitive `Answer:` prefix in the text being graded. EM conditional on
  compliance uses a different subset for each model and is not a comparison of
  reasoning ability on identical questions.
- Format compliance means the final `format` metric equals 1; it is not identical
  to normal answer termination. In q35-fast, `nq/test_1028` wrote `Answer:` in the
  fifth assistant message and then called search. It ended with `max_turns`, but
  had format 1, EM 0 and reward 0. Thus 381 records passed the format check and 380
  ended as `answered`. The counts coincide in the other four runs.
- Wall-clock evaluation time includes generation, prefill, retrieval and
  scheduling. It is not a training-cost estimate.
- The turn-limit column below counts `termination=max_turns`, excluding successful
  fifth-turn answers. For example, q35-fast has 271 five-turn trajectories, of which
  242 ended because of the turn limit.
- oMLX uses a text interface and the client retokenizes responses. Recorded token
  count mismatches were 3/2,469 sampling calls for q35-original, 3/2,291 for q35-fast,
  4/2,269 for q35-think-fast, 1/1,225 for q3i-original and 1/1,186 for q3i-fast.
  These checks compare counts, not token identities. Full per-turn token IDs were
  not saved, so the records do not support exact token replay.

## Overall results

| Configuration | EM | F1 | Format compliance | EM given compliance | Turn-limit termination | Mean generated tokens/question | Time (minutes) |
|---|---:|---:|---:|---:|---:|---:|---:|
| q35-original | 32.6% | 38.9% | 53.7% | 60.8% | 43.6% | 327 | 41.6 |
| q35-fast | 33.3% | 40.8% | 60.4% | 55.1% | 38.4% | 256 | 34.4 |
| q35-think-fast | 37.2% | 44.9% | 63.9% | 58.3% | 35.8% | 591 | 66.3 |
| q3i-original | 23.9% | 32.0% | 91.1% | 26.3% | 0.5% | 156 | 14.6 |
| q3i-fast | 20.4% | 26.1% | 83.0% | 24.6% | 0.0% | 79 | 10.2 |

These results favor Qwen3.5 as the candidate for subsequent work. q35-fast is the
practical starting point for local iteration; q35-think-fast achieved the highest
accuracy in this comparison. q35-fast answered only four more questions correctly
than q35-original, providing no clear accuracy improvement, while generating about
22% fewer tokens and taking about 17% less time. Enabling thinking added 25 correct
answers over q35-fast (+3.96 percentage points), at about 2.31 times the generated
tokens and 1.93 times the elapsed time. Only 2/631 thinking trajectories ended with
a truncated turn, so the 1,024-token generation cap was not the dominant recorded
failure mode.

Approximate 95% intervals for paired EM differences use the standard error of
per-question differences. They are not adjusted for multiple comparisons and do
not include uncertainty from rerunning stochastic generation.

| Comparison (first minus second) | EM difference (percentage points) | Approximate interval | First-only / second-only correct |
|---|---:|---:|---:|
| q35-fast − q35-original | +0.63 | −2.38 to +3.65 | 49 / 45 |
| q35-think-fast − q35-fast | +3.96 | +0.92 to +7.01 | 61 / 36 |
| q3i-fast − q3i-original | −3.49 | −6.49 to −0.48 | 36 / 58 |
| q35-fast − q3i-fast | +12.84 | +9.12 to +16.55 | 117 / 36 |
| q35-original − q3i-original | +8.72 | +5.04 to +12.40 | 100 / 45 |

## Turn limits and retrieval behavior

The three Qwen3.5 runs had 275, 242 and 226 turn-limit terminations, respectively.
Tool-argument errors occurred in only 25, 24 and 21 trajectories. None of the
turn-limit trajectories had errors on every tool call. Invalid tool arguments
therefore do not explain most failures to answer.

| Configuration | Turn-limit terminations | Eligible after excluding gold-in-question cases | Gold string in retrieved documents | Found within first four tool responses |
|---|---:|---:|---:|---:|
| q35-original | 275 | 248 | 108 | 100 |
| q35-fast | 242 | 217 | 84 | 73 |
| q35-think-fast | 226 | 198 | 79 | 70 |

These counts come from the original offline string-matching diagnostics. Echoed
`Query:` lines are removed before matching normalized documents against normalized
gold answers. `diagnose_run.py` increments its position for each tool response,
not each assistant turn. A turn can call multiple tools, so these positions must
not be interpreted as assistant-turn numbers.

Two limitations are essential:

1. A gold string appearing in a document does not establish the required entity
   relationship. For example, q35-fast's `bamboogle/test_20` asks which city a
   Philippine politician served as mayor. Documents mentioning `Quezon City`
   discuss restaurants and garrisons; the match does not prove sufficient evidence.
2. A search executed on the fifth assistant turn leaves no subsequent turn to
   answer. It cannot be counted as the model seeing evidence and then choosing to
   continue searching.

The diagnostic script's hypothetical EM values of 49.8%, 46.6% and 49.8% assume
every selected string-match case becomes correct. They are neither supported
accuracy predictions nor overall capability upper bounds. A missing gold string
also does not prove that the answer could not be inferred.

Qwen3-Instruct had no recorded search calls in 163/631 original-prompt trajectories
(25.8%) and 123/631 fast-prompt trajectories (19.5%); all Qwen3.5 runs were below 1%.
Qwen3-Instruct was faster but used retrieval less often. Its higher format
compliance does not establish stronger search ability.

## Answer-format sensitivity for Qwen3-Instruct

Of q3i-fast's 99 `no_valid_answer` records, 98 lacked `Answer:` and one contained
multiple prefixes. Of q3i-original's 49 such records, 43 lacked the prefix and six
contained multiple prefixes.

The original offline sensitivity analysis considered only `no_valid_answer`
records: use the complete final text when the prefix is missing, or the text after
the last prefix when it appears multiple times, then still require normalized
exact match. This recovers 28 q3i-fast answers, raising hypothetical EM from 20.4%
to 24.9%, and four q3i-original answers, from 23.9% to 24.6%. It recovers one
q35-original answer and none for q35-fast or q35-think-fast.

The official scores are unchanged. The exercise shows that much of q3i-fast's
decline involves answer formatting and should not be read directly as worse
reasoning. Even under this relaxed extraction, Qwen3.5-fast remains substantially
ahead of Qwen3-Instruct-fast.

This analysis covers normally completed outputs classified as `no_valid_answer`.
It says nothing about potential answers in turn-limited or truncated trajectories.
The Qwen3.5 runs had 11, seven and zero `no_valid_answer` records, with one, zero
and zero recovered matches, respectively.

Training and local evaluation use `search_r1.tools.TextAnswerReward`. The saved
evaluation configurations use `F1 + 0.1*(format-1) + 0.1*(searched-1)`. F1 and EM
share `_extract_answer`: missing or repeated case-sensitive `Answer:` prefixes
make both scores zero, rather than merely incurring the 0.1 format penalty.
Environment-level early termination, including parsing failures and truncated
turns, can apply a failure reward without ordinary answer grading. All recorded
`max_turns` trajectories had EM 0. The q35-fast exception noted above had reward 0;
the other turn-limit records had reward −0.1.

## EM by source

| Source | q35-original | q35-fast | q35-think-fast | q3i-original | q3i-fast |
|---|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 33.0% | 31.0% | 31.0% | 27.0% | 25.0% |
| bamboogle (31 questions) | 19.4% | 22.6% | 22.6% | 29.0% | 16.1% |
| hotpotqa | 36.0% | 35.0% | 43.0% | 30.0% | 16.0% |
| musique | 9.0% | 12.0% | 12.0% | 8.0% | 7.0% |
| nq | 33.0% | 33.0% | 37.0% | 19.0% | 20.0% |
| popqa | 37.0% | 38.0% | 43.0% | 21.0% | 22.0% |
| triviaqa | 52.0% | 54.0% | 62.0% | 37.0% | 34.0% |

The thinking run's gains over q35-fast were concentrated in HotpotQA, NQ, PopQA
and TriviaQA; it did not improve Musique, 2Wiki or Bamboogle in this run. Per-source
sample sizes are small, especially Bamboogle, so small differences do not support
strong rankings.

## Model choice and scope of subsequent work

- Preserve all five runs as historical baselines and continue with Qwen3.5 as the
  main candidate.
- A prompt allowing at most four search turns and requiring an answer on turn five
  was an idea in the original analysis. It was not run, is not a demonstrated
  improvement, and is not an additional test scheduled by this report.
- Increasing the turn cap or relaxing scoring would change the comparison.
  Formal baselines and trained models must use the same conditions.
- Retain q35-think-fast as an accuracy reference, with its higher generation cost.
- Future formal training and baselines must use consistent renderer, prompt and
  scoring settings. These local selection results cannot be combined with cloud
  results under a new renderer to claim a training gain.

## Additional analysis: the search-strategy gap

The two fast-prompt runs differ substantially in search behavior:

| Metric | q35-fast | q3i-fast |
|---|---:|---:|
| No recorded search calls | 3/631 (0.5%) | 123/631 (19.5%) |
| Exactly one search call | 193/631 (30.6%) | 470/631 (74.5%) |
| At least two search calls | 435/631 (68.9%) | 38/631 (6.0%) |
| Mean search calls/question | 3.01 | 0.88 |
| Mean query strings/question | 6.02 | 1.03 |

No recorded calls includes attempted calls that failed to parse; it does not always
mean the model deliberately avoided search. A single call may contain multiple
queries. About 94% of q3i-fast questions stopped at zero or one recorded call,
providing stronger evidence of limited multi-step search than format compliance
alone.

To reduce differences in question selection and non-searching cases, the original
analysis compared the same 458 questions where both models searched and no gold
answer was already present in the question:

| Check | q35-fast | q3i-fast |
|---|---:|---:|
| Gold string in first tool response | 162/458 (35.4%) | 138/458 (30.1%) |
| Gold string in any tool response | 267/458 (58.3%) | 145/458 (31.7%) |

Both models produced some first-response string matches. Later responses covered
105 additional questions for q35-fast and seven for q3i-fast. The larger difference
appeared in subsequent coverage rather than the first response alone. String
matches still do not establish sufficient evidence, and the models used different
amounts of their search budgets. This is not a causal decomposition or a comparison
of the retrieval engine's intrinsic quality.

The original-prompt runs show the same pattern: among 415 eligible questions where
both searched, first responses matched 149 versus 129 questions, and any response
matched 266 versus 144.

For example, `2wikimultihopqa/dev_8136` asks when Casimir I of Kuyavia's father died.
q3i-fast searched for the father and family history, identified Konrad I of Masovia,
then answered `Unknown` because the documents did not give a death date. q35-fast
subsequently searched `Konrad I of Masovia death date` and answered `31 August 1247`.
This illustrates a failure to use a newly identified entity for a follow-up query.

Of the 117 questions q35-fast got right and q3i-fast got wrong, q3i-fast made zero
calls on 18, one call on 95 and at least two calls on four. This supports insufficient
search or early stopping as an important contributor, but does not prove that
forcing more calls would produce correct answers. Qwen3.5 itself often searched
too long without answering.

The strict fast-prompt EM gap is 12.84 percentage points. Relaxing only the answer
prefix check described above raises q3i-fast to 157/631 (24.9%), still 8.4 points
below q35-fast's 33.3%. Formatting explains only part of the gap. Query coverage,
multi-hop entity tracking and evidence use are plausible contributors; these
observations do not identify their independent causal effects.
