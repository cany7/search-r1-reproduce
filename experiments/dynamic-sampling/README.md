# Dynamic sampling

Training continued from the main reproduction's step-100 checkpoint. Question
groups whose trajectories all received the same reward were filtered out, and
new questions were drawn to refill the training batch. Five updates were
completed (100 → 105) under the limited-search rule, and the step-105
checkpoint was evaluated on the development set.

[Training records](training/) · [Step-105 dev evaluation](evaluations/limited-search/dev/step-0105/) · [Restart validation](smoke/)

[Experiment metadata](experiment.json)
