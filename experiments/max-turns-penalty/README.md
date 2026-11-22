# Max-turns unanswered penalty

Training continued from the main reproduction's step-75 checkpoint to step 100.
Episodes that reached the five-turn cap without a valid answer received a final
training reward of −0.3 instead of −0.1. The control is the main reproduction's
extended training from step 75 to step 100, which used the original reward.

Training used the limited-search rule. Evaluation does not apply this penalty
and keeps the base scoring.

[Training records](training/) · [Step-100 dev evaluation](evaluations/limited-search/dev/step-0100/) · [Step-100 test evaluation](evaluations/limited-search/test/step-0100/) · [Control training](../reproduce-main/training/extended-training/)

[Experiment metadata](experiment.json)
