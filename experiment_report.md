# Search-R1 reproduction report

[简体中文](docs/experiment_report_zh.md)

We trained Qwen3.5-4B on one RTX PRO 6000 Blackwell GPU (96 GB) to answer open-domain questions with access to Wikipedia. The model had five turns per question. It could answer immediately, search and read the results before answering, or spend all five turns searching. A search on the fifth turn left no opportunity to use the results.

On the 1,894-question test set, the base model answered 623 questions correctly (32.89% exact match). The checkpoint we selected, at step 75, answered 892 correctly (47.10%) and gave a valid answer on 99.21% of questions. An earlier checkpoint at step 50 had the highest exact match (48.15%), but its search calls returned so much text that 160 questions exceeded the context budget. By step 75, a limit of three queries per turn and another 25 training updates had brought context overflows to zero. With further training, the model more often spent all five turns searching. At step 100, it ran out of turns on 234 test questions, up from 8 at step 75.

The task and data come from Search-R1. We took the fast prompt, the F1+ reward and the REINFORCE training method from Search-R1++, and adapted the training loop and tool interface from the Tinker Cookbook search recipe. The archive includes configurations, per-dataset evaluation summaries and individual trajectories. [Search-R1](https://arxiv.org/html/2503.09516v4) · [Search-R1++](https://arxiv.org/html/2602.19526v1) · [Tinker Cookbook recipe](https://github.com/thinking-machines-lab/tinker-cookbook/blob/1991897cba59ea9d9b0030d183d1867b28eba9d9/tinker_cookbook/recipes/search_tool/README.md) · [Experiment archive](experiments/README.md) · [Training logs](https://forge.coreweave.com/wandb/timye7-personal/search-r1-reproduce?nw=sw0myi4atxa)

## Test results

Exact match (EM) counts a question as correct if the normalized answer matches any reference answer. F1 measures token overlap. The answer rate is the fraction of questions for which the scorer could extract a valid `Answer:` field. "EM when answered" uses only those questions, so its denominator varies by checkpoint. The last two columns count, respectively, questions that ended at the turn limit without a valid answer and questions that exceeded the input context budget.

| Checkpoint and search rule | Correct | EM | F1 | Answer rate | EM when answered | Search calls per question | Unanswered at turn limit | Context overflows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base, original | 623 | 32.89% | 39.59% | 59.93% | 54.89% | 3.04 | 714 | 0 |
| Step 25, original | 852 | 44.98% | 53.81% | 97.78% | 46.00% | 1.62 | 11 | 0 |
| Step 50, original | 912 | 48.15% | 57.03% | 88.97% | 54.12% | 2.20 | 29 | 160 |
| **Step 75, limited (selected)** | **892** | **47.10%** | **55.99%** | **99.21%** | **47.47%** | **2.20** | **8** | **0** |
| Step 100, limited | 876 | 46.25% | 55.37% | 86.43% | 53.51% | 2.91 | 234 | 0 |
| Step 100, limited, larger penalty | 849 | 44.83% | 53.09% | 83.05% | 53.97% | 3.24 | 319 | 0 |

Steps 50 and 75 both averaged 2.20 attempted search calls per test question. A call at step 50 could contain a long list of queries, each returning up to three documents. At step 75, each turn could return at most nine document entries. Sampling the same test set took 40.2 minutes at step 50 and 15.7 minutes at step 75. Both the weights and the search rule changed between those evaluations. To measure the immediate effect of the rule change, we also evaluated the same step 50 weights under both rules on the development set. We evaluated the base model only under the original rule. The earliest checkpoint evaluated with the limit was step 25 on the development set. [Original search evaluations](experiments/reproduce-main/evaluations/original-search/test/) · [Limited search evaluations](experiments/reproduce-main/evaluations/limited-search/test/)

## Experimental setup

Training questions came from Natural Questions (NQ) and HotpotQA. We used fixed development and test sets of 631 and 1,894 questions respectively, with no questions in common. Both sets include the two training sources and five other datasets. Each checkpoint produced one trajectory per question in each evaluation setting. Overall scores give each question equal weight. On the development set, the standard error of a single checkpoint's overall EM is about two percentage points. For comparisons between checkpoints, we also paired results by dataset and question ID. [Evaluation sets](eval_sets/) · [Sampling script](scripts/make_eval_sets.py) · [Evaluation code](search_r1/offline_eval.py)

### Training runs

| Run | Starting point | Updates | Change |
| --- | --- | ---: | --- |
| Original search | Base model | 50 | Execute all queries in a search call. |
| Limited search | Step 50 | 25, then 25 more | Execute the first three valid queries in the first search call of each turn. |
| Dynamic sampling | Step 100 | 5 | Replace questions whose eight trajectories all receive the same reward. |
| Larger penalty | Step 75 | 25 | Set the reward for exhausting all five turns without a valid answer to −0.3, down from −0.1. |

Each update in the main run used 64 questions and eight trajectories per question. The first 100 updates used 51,200 trajectories in total. We disabled native thinking and trained rank-32 LoRA adapters with Adam at a learning rate of `4e-5`. [Training configuration](experiments/reproduce-main/training/stage-1/config.json) · [Parameter update](search_r1/training/update.py)

### Rollouts and retrieval

On each turn, the model generated one message. A `search` call retrieved up to three documents per executed query from a BM25 index of the 2018 Wikipedia corpus. The environment appended the results to the conversation before the next turn. A message without a tool call ended the trajectory and was scored. A trajectory could also end at the five-turn limit, on context overflow, on truncated generation or on a parsing error. Unanswered trajectories still contributed to training.

The fast prompt showed examples of tool calls and answers. Sampling used a temperature of 1.0 and a maximum of 1,024 generated tokens per turn. Before each generation, the environment checked the input against a budget of 32,768 tokens. We kept the weights fixed while sampling trajectories concurrently. Once all trajectories finished, the trainer calculated rewards and advantages, updated the model once and sent the new weights to the sampler. The dynamic sampling branch used the same weights for all replacement questions within an update. [Training loop](search_r1/training/loop.py) · [Group sampling](search_r1/training/sampling.py) · [Tool environment](search_r1/env/agent_tool_env.py) · [Context budget](search_r1/env/message_env.py)

### Reward and policy update

For trajectory $i$ of question $q$, the standard F1+ reward was

$$
R_{q,i}
= \max_{y\in Y_q}\operatorname{F1}(\hat y_{q,i},y)
-0.1\,\mathbf{1}[\text{no search call}]
-0.1\,\mathbf{1}[\text{no valid answer}].
$$

Here $Y_q$ is the set of reference answers. The scorer extracted $\hat y_{q,i}$ from the final model message and calculated F1 on normalized tokens. If no answer could be extracted, F1 was zero. Calling search once was enough to avoid the search penalty, regardless of how many queries the call contained. Trajectories ending in context overflow, truncated generation or a parsing error received a fixed reward of −0.1. The larger penalty branch used a separate terminal reward of −0.3 for trajectories that exhausted all five turns without a valid answer. [Reward code](search_r1/env/tools.py) · [Early termination](search_r1/env/message_env.py) · [Turn limit penalty](search_r1/env/search_tool_env.py)

We used REINFORCE with importance sampling. The baseline for each trajectory was the mean reward of the eight trajectories for the same question. All tokens generated by the model in that trajectory shared the advantage

$$
A_{q,i}=R_{q,i}-\frac{1}{8}\sum_{j=1}^{8} R_{q,j}.
$$

Subtracting the mean gives zero advantage to all eight trajectories when their rewards are equal. We used the centered rewards directly, without dividing by the group standard deviation or fitting a value network. For a batch of questions $\mathcal B$, the loss was

$$
\mathcal L(\theta)
=-\sum_{q\in\mathcal B}\sum_{i=1}^{8}
\sum_{t\in\mathcal G_{q,i}}
\frac{\pi_\theta(a_t\mid h_t)}{\pi_{\mathrm{roll}}(a_t\mid h_t)}
A_{q,i}.
$$

$\mathcal G_{q,i}$ contains the model-generated token positions. For each sampled token $a_t$, the history $h_t$ includes the question, preceding messages and search results. The sampler recorded the token's log probability under the rollout policy $\pi_{\mathrm{roll}}$. During the update, the trainer used that value to compute the ratio of the token's probability under the current policy to its probability under the rollout policy. Prompt and tool tokens received zero advantage. Training used neither PPO clipping nor an additional KL penalty. [Advantages and token masks](search_r1/training/data.py) · [Loss configuration](search_r1/training/update.py) · [SkyRL loss implementation](https://github.com/NovaSky-AI/SkyRL/blob/79d4c8946bba23c2e0617e96ac1095c596093d83/skyrl/backends/skyrl_train/utils/ppo_utils.py)

### Comparison with the reference implementations

The closest implementation is the Tinker Cookbook recipe. It uses the same Qwen3.5-4B model with native thinking disabled, rank-32 LoRA and eight trajectories per question. Training scale, retrieval and scoring differ across the projects.

| Project | Training scale | Method |
| --- | --- | --- |
| Tinker Cookbook | 512 questions per default update, eight times our batch size. Its 30-step validation run sampled about 123,000 trajectories. | Dense retrieval and an exact-match reward with a penalty for invalid answer format. |
| Search-R1 | Qwen2.5-3B and 7B on eight H100 GPUs, 500 steps for each algorithm. | PPO or GRPO. |
| Search-R1++ | Qwen2.5-3B and 7B on eight A100 GPUs, 600 steps. | Its REINFORCE configuration used five trajectories per question and KL regularization. |

The Cookbook reports that its training reward rose from 0.39 to 0.60 in 30 steps. Our reward was 0.46 at step 1 and 0.56 at step 30, with a mean of 0.59 over steps 26 to 30. F1+ gives partial credit for answer overlap and penalizes skipping search, so these reward curves are only roughly comparable. We therefore measured training gains on our own fixed evaluation sets. We used the fast prompt, F1+ and REINFORCE together and did not measure their individual contributions. [Cookbook training configuration](https://github.com/thinking-machines-lab/tinker-cookbook/blob/1991897cba59ea9d9b0030d183d1867b28eba9d9/tinker_cookbook/recipes/search_tool/train.py) · [Cookbook results](https://github.com/thinking-machines-lab/tinker-cookbook/blob/1991897cba59ea9d9b0030d183d1867b28eba9d9/tinker_cookbook/recipes/search_tool/README.md) · [Training metrics](experiments/reproduce-main/training/stage-1/metrics.jsonl)

## Model and prompt selection

Before training, we evaluated two models locally with oMLX on the same 631 development questions. The original Cookbook prompt asked for step-by-step reasoning. The fast prompt omitted that instruction and illustrated tool calls and answers directly.

| Local model | Prompt | EM | Answer rate | Questions ending at turn limit | Search calls per question | Generated tokens per question |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-4B-Instruct-2507 | Original | 23.93% | 91.13% | 3 | 0.96 | 156 |
| Qwen3-4B-Instruct-2507 | Fast | 20.44% | 83.04% | 0 | 0.88 | 79 |
| Qwen3.5-4B, thinking off | Original | 32.65% | 53.72% | 275 | 3.35 | 327 |
| **Qwen3.5-4B, thinking off** | **Fast** | **33.28%** | **60.38%** | **242** | **3.01** | **256** |
| Qwen3.5-4B, thinking on | Fast | 37.24% | 63.87% | 226 | 2.96 | 591 |

With the fast prompt, Qwen3-4B-Instruct-2507 (Qwen3-Instruct below) searched at least twice on 38 questions. Qwen3.5 with thinking disabled did so on 435 and reached the turn limit on 242. We chose Qwen3.5 because it already used intermediate results to search again, leaving room to study whether training could improve its decision about when to stop searching. Enabling native thinking yielded 25 more correct answers but increased generated tokens by a factor of 2.31. Given the single-GPU budget, we used the fast prompt with thinking disabled. The local runs were used only to select the model and prompt. The base model results elsewhere in this report come from the cloud evaluation. [Local evaluations and trajectories](experiments/local-baselines/README.md)

We first tested tool calls and parameter updates on an A100 with 80 GB of memory. After a restart, we verified that we could load a full checkpoint for evaluation and resume training with the saved optimizer state. Training and evaluation for the reported runs used a self-hosted SkyRL backend through the Tinker SDK. [Cloud validation](experiments/cloud-validation/README.md) · [Training environment](experiments/formal-environment.txt)

## The first 50 updates

The base model, step 25 and step 50 answered 623, 852 and 912 test questions correctly. Pairing the results shows how the gains changed over training. For each pair of checkpoints, we counted the net change in correct answers separately for questions both checkpoints answered and questions only one answered.

| Test comparison | Net gain in correct answers | Both checkpoints answered | Only one checkpoint answered |
| --- | ---: | ---: | ---: |
| Base to step 25 | +229 | +21 | +208 |
| Step 25 to step 50 | +60 | +78 | −18 |

In the first 25 steps, the answer rate rose from 59.93% to 97.78%. Questions answered by only one checkpoint accounted for 208 of the 229 additional correct answers. Over the next 25 steps, the answer rate fell to 88.97%. On the 1,657 questions answered by both checkpoints, step 50 answered 78 more correctly. That gain was large enough for overall EM to rise even as more questions went unanswered. [Test summaries and trajectories](experiments/reproduce-main/evaluations/original-search/test/)

### Query growth under the original search rule

The original tool accepted a list of queries and appended the results from each query to the context. An additional query gave the model another chance to retrieve an answer but cost nothing in the reward. As query lists grew, both the input to the model and the time per update increased.

| Training metric | Step 1 | Step 45 |
| --- | ---: | ---: |
| Mean queries per search call | 1.9 | 14.9 |
| Estimated document entries per call | 6 | 45 |
| Mean input tokens on the final turn | 3,405 | 11,989 |
| Minutes per update, including sampling | 9.4 | 33.6 |

The document counts assume three results per query and can include duplicates. Most of the input growth came from retrieval. At step 36, the final turn averaged 9,290 input tokens, an estimated 8,881 of which came from the prompt and retrieved text. An update averaged 37.1 minutes over steps 46 to 50. In the step 50 test evaluation, 160 of 1,894 questions (8.45%) ended in context overflow. [Training metrics](experiments/reproduce-main/training/stage-1/metrics.jsonl) · [Step 50 test summary](experiments/reproduce-main/evaluations/original-search/test/step-0050/summary.json)

To choose a query limit, we took the 308 development questions the step 50 model had answered correctly. For each successful search call in the recorded trajectories, we kept the documents from the first one, three or five queries and checked for a reference answer or alias. The retained text contained an answer for 230, 273 and 286 questions respectively. Allowing three queries instead of one covered 43 more questions. Increasing the limit from three to five covered another 13. This replay measured only whether an answer appeared in the retrieved documents. Later queries in each record had been written by a model that saw the full results, so estimating accuracy under the new rule required fresh rollouts. [Step 50 development trajectories](experiments/reproduce-main/evaluations/original-search/dev/step-0050/trajectories.jsonl)

We resumed from the full step 50 checkpoint with a limit of three valid queries from the first search call on each turn. That allowed up to nine document entries per turn and left the model free to search again after reading them. We removed the statement in the tool description that every query would return results. We considered announcing the limit in the prompt or flagging dropped queries in the tool response, but instead kept the prompt unchanged and returned only the executed queries' results, leaving the model to adapt through the search results and the existing reward alone. [Search tool implementation](search_r1/env/search_tool_env.py)

## Training with a three-query limit

The first two rows use identical model weights. The next two show what happened when training continued with the limit in place.

| Development set (631 questions) | EM | Answer rate | Context overflows | Unanswered at turn limit |
| --- | ---: | ---: | ---: | ---: |
| Step 50, original search | 48.81% | 89.70% | 51 | 9 |
| Step 50, limited search | 45.01% | 80.67% | 0 | 116 |
| Step 75, limited search | 47.39% | 98.89% | 0 | 5 |
| Step 100, limited search | 47.07% | 87.80% | 0 | 69 |

With the limit in place, the step 50 model still requested a median of 16 queries on the first turn. The tool executed at most three. Context overflows fell from 51 to 0, but 116 questions (18.4%) now exhausted all five turns without an answer. Applying the same rule change to the step 25 weights also reduced development EM, from 44.37% to 41.52%. [Original step 50 results](experiments/reproduce-main/evaluations/original-search/dev/step-0050/) · [Limited search evaluations](experiments/reproduce-main/evaluations/limited-search/dev/)

By step 75, only five development questions went unanswered at the turn limit. The model answered 15 more correctly than at step 50 under the same rule. Updates also became faster. An update averaged 11.4 minutes over steps 51 to 55 and 9.3 minutes over steps 71 to 75, down from 37.1 minutes in the final five steps under the original rule. [Steps 51 to 75](experiments/reproduce-main/training/stage-2/metrics.jsonl)

From step 75 to step 100, the model requested fewer queries per call but made more calls per question. On the test set, the median number of queries requested on the first turn fell from 10 to 5, and mean calls per question rose from 2.20 to 2.91. The number of questions left unanswered at the turn limit rose from 8 to 234. We selected step 75 after completing both the development and test evaluations. Steps 75 and 100 differed by only two correct answers on the development set (299 and 297), but the answer rate at step 75 was 11.09 percentage points higher. [Main run archive](experiments/reproduce-main/README.md) · [Limited search test results](experiments/reproduce-main/evaluations/limited-search/test/)

A growing share of training questions also produced identical rewards across all eight trajectories. These groups supplied zero advantage to the update. They accounted for an average of 37.7 of 64 groups per step (58.9%) over steps 51 to 75, rising to 43.3 (67.6%) over steps 76 to 100. This prompted a separate run to test whether filling each batch with questions whose trajectories received different rewards would improve the answer rate. We kept the reward and prompt fixed in that run, including the existing −0.1 reward for exhausting the turns without an answer. [Training records](experiments/reproduce-main/training/) · [Advantage calculation](search_r1/training/data.py)

## Follow-up experiments

### Dynamic sampling from step 100

For each update, we held the policy fixed and drew fresh questions to replace groups in which all eight rewards were equal. Sampling stopped once 64 groups with some reward variation were available. The percentages below use all attempted groups as the denominator and include any qualifying groups sampled beyond the required 64 in the final round.

| Step | Questions sampled | Groups with reward variation | Minutes per update |
| ---: | ---: | ---: | ---: |
| 101 | 158 | 41.1% | 21.8 |
| 102 | 187 | 34.8% | 23.1 |
| 103 | 192 | 33.9% | 23.7 |
| 104 | 202 | 32.7% | 26.3 |
| 105 | 241 | 29.9% | 27.8 |

Every update reached 64 qualifying groups. By the fifth update, the sampler needed 241 questions to fill a batch, up from 158 on the first. We stopped this branch at step 105 and evaluated it only on the development set. Questions left unanswered at the turn limit fell from 69 at step 100 to 56 at step 105. For comparison, step 75 left only five. [Dynamic sampling configuration and metrics](experiments/dynamic-sampling/training/) · [Step 105 evaluation](experiments/dynamic-sampling/evaluations/limited-search/dev/step-0105/)

Development EM rose from 47.07% to 47.70% (297 to 301 correct), and F1 rose from 56.53% to 57.00%. On the 530 questions both checkpoints answered, each got 288 right. Step 105 answered 13 correctly among the questions it alone answered, compared with 9 for step 100. Across all 631 questions, 32 changed from incorrect to correct and 28 from correct to incorrect. The paired EM difference was +0.63 percentage points, with an approximate 95% interval of −1.77 to +3.04 points. This single evaluation leaves both the size and the direction of the difference uncertain. [Step 100 trajectories](experiments/reproduce-main/evaluations/limited-search/dev/step-0100/trajectories.jsonl) · [Step 105 trajectories](experiments/dynamic-sampling/evaluations/limited-search/dev/step-0105/trajectories.jsonl)

### A larger penalty for running out of turns

We restored the model and optimizer from step 75 and ran 25 ordinary updates in the same question order as the main run. The only training change was a terminal reward of −0.3 for using all five turns without a valid answer. A valid answer on turn five received the usual score. Both branches used the same scoring procedure for development and test evaluations. [Branch configuration and evaluations](experiments/max-turns-penalty/README.md)

At step 100, the penalty branch answered 286 development questions correctly, 11 fewer than the main run. On the test set it answered 849 correctly, 27 fewer. The larger penalty also produced more unanswered questions at the turn limit in this run (319 versus 234). The median number of queries requested on the first turn was 30 in the penalty branch and five in the main run. Of the 1,479 test questions both checkpoints answered, the main run got 829 right and the penalty branch got 828 right. The remaining 26-answer gap came from questions that only one of them answered. [Main run trajectories](experiments/reproduce-main/evaluations/limited-search/test/step-0100/trajectories.jsonl) · [Penalty branch trajectories](experiments/max-turns-penalty/evaluations/limited-search/test/step-0100/trajectories.jsonl)

To measure how often the larger penalty changed the training signal, we re-scored the branch's recorded trajectories with the −0.1 penalty and recalculated each group's advantages. Of 12,800 trajectories, 159 triggered the larger penalty, including 149 in the last five steps. Only 51 of 1,600 question groups (3.2%) had different advantages under the two reward settings. For 135 trajectories, the larger penalty made an already negative advantage more negative. The other 24 belonged to three groups in which every trajectory ran out of turns; because all eight rewards dropped by the same amount, their advantages stayed at zero. The penalty applied only at termination and placed no direct cost on submitting more queries. [Penalty branch metrics](experiments/max-turns-penalty/training/metrics.jsonl) · [Terminal reward condition](search_r1/env/search_tool_env.py)

## Two search trajectories

**Finding a date through a second search.** Question `dev_8136` from 2WikiMultihopQA asks when the father of Casimir I of Kuyavia died. In the local fast-prompt evaluations, both base models found his father's name, Konrad I of Masovia, on the first turn, but the retrieved text did not give his death date. Qwen3-Instruct answered `Unknown`. Qwen3.5 searched for `Konrad I of Masovia death date`, retrieved the date and answered correctly on turn three with August 31, 1247. [Qwen3-Instruct trajectory](experiments/local-baselines/evaluations/base-q3i-fast-n100/trajectories.jsonl) · [Qwen3.5 trajectory](experiments/local-baselines/evaluations/base-q35-fast-n100/trajectories.jsonl)

**Using all five turns on the wrong leads.** Test-set question `dev_560` from MuSiQue (the ID is inherited from the source data) asks which child of the author of the memoir *One More Time* worked on its stage adaptation. The step 75 model missed the memoir on the first turn. One of its executed queries on the second turn, `One More Time book`, retrieved an entry saying that Carol Burnett and her daughter Carrie Hamilton wrote *Hollywood Arms* together. The model answered Carrie Hamilton on turn three. The step 100 model requested five queries on its first turn, then pursued leads including Elizabeth Gilbert, T. J. Clark and John Steinbeck. It used every turn for search and gave no answer. [Step 75 trajectory](experiments/reproduce-main/evaluations/limited-search/test/step-0075/trajectories.jsonl) · [Step 100 trajectory](experiments/reproduce-main/evaluations/limited-search/test/step-0100/trajectories.jsonl)

## Discussion

The answer rate explains much of the early gain in exact match. It also helps interpret the step 75 selection. At that checkpoint, nearly every question received an answer, the context budget held throughout evaluation, and test EM was about one percentage point below its peak at step 50. Extending the same run to step 100 yielded almost the same development EM, even though the model ran out of turns much more often. Tracking both queries per call and calls per question made the change visible. The first measure showed how quickly retrieved text could fill the context, while the second showed how much of the five-turn budget the model spent searching.

Neither follow-up established a reliable improvement over step 75. Dynamic sampling provided nonzero advantages in every group used for an update, but the four additional correct development answers relative to step 100 fall within the uncertainty of the paired comparison. The larger penalty changed the advantages in only 3.2% of sampled groups, and that branch finished with more unanswered test questions than the main run.

## Appendix

### Test exact match by dataset

NQ and HotpotQA supplied the training questions. The remaining five datasets test questions from sources not used in training. Bamboogle has 94 test questions, so one correct answer changes its EM by about 1.06 percentage points.

| Dataset | Test questions | Base model | Step 50, original | Step 75, limited |
| --- | ---: | ---: | ---: | ---: |
| NQ | 300 | 27.00% | 38.33% | 36.67% |
| HotpotQA | 300 | 37.00% | 51.00% | 51.00% |
| TriviaQA | 300 | 53.67% | 68.67% | 67.67% |
| PopQA | 300 | 32.67% | 48.33% | 43.67% |
| 2WikiMultihopQA | 300 | 36.00% | 54.67% | 62.67% |
| MuSiQue | 300 | 10.00% | 25.00% | 19.67% |
| Bamboogle | 94 | 36.17% | 57.45% | 51.06% |

### Runtime and checkpoint recovery

In the preliminary A100 run, the server trimmed left padding shared across a microbatch. With a microbatch size of one, we completed three updates and resumed training after a restart. Evaluating the same 271 questions took about 84 seconds at a concurrency of 32 and 87 seconds at a concurrency of 64. We used 32 for the reported evaluations. Training enabled gradient checkpointing for sequences longer than 8,000 tokens and saved training state every five steps. Each branch restored both the model and optimizer from its parent checkpoint. [Validation and timing](experiments/cloud-validation/README.md) · [Environment and software versions](experiments/formal-environment.txt) · [Deployment and recovery scripts](scripts/README.md)
