# MISQ-HF

Effective decision-making and problem-solving in conversational systems require the ability to identify and acquire missing information through targeted questioning. A key challenge lies in efficiently narrowing down a large space of possible outcomes by posing questions that minimize uncertainty. To address this, we introduce a novel framework that leverages Large Language Models (LLMs) to generate information-seeking questions, with Monte Carlo Tree Search (MCTS) to strategically select questions that maximize information gain, as a part of inference-time planning. Our primary contribution includes a hierarchical feedback mechanism that exploits past interaction patterns to guide future strategy. Specifically, each new problem is mapped to a cluster based on semantic similarity, and our UCT (Upper Confidence bound for Trees) formulation employs a cluster-specific bonus reward to prioritize successful question trajectories that have proven effective for similar problems in the past. 

## Setup

1. Install `misq` package
```bash
git clone https://github.com/[user-name]/[repo].git MISQ
cd MISQ
conda create --name misq python=3.13
conda activate misq
pip install -r requirements.txt 
pip install -e .
```

2. Set up AWS Bedrock using the `aws configure` command.
   
3. install dataset [here](https://drive.google.com/drive/folders/1QhhsPinylvbgm52zX4VjwiKDxAgPvyVR?usp=sharing) and put files under `src/misq/data/`


## Use
Run experiments via `run.py`, which implements the MISQ algorithm, as well as the naive prompting method. Arguments are as follows: [note multiple other MCTS-related argumemts are present in run.py]

- `--guesser_model`: The name of model used to plan and ask questions

- `--examiner_model`: The name of model used to provide environment feedback. Fixed to be `gpt-4` currently.

- `--task` and `--dataset`: Select the corresponding task name and dataset according to the table below.

    | Description       | task  | dataset               |
    |-------------------|-------|-----------------------|
    | 20 Question Game  | `20q` | `thing` / `common` |
    | Medical Diagnosis | `md`  | `DX` / `MedDG`        |
    | Troubleshooting   | `tb`  | `FloDial`             |

- `--desc`: A short description or name of the experiment to store in logs and roots directory

- `--feedback_upd`: If True, use feedback and update bonus reward on success

- `--bonus_factor`: Proportion of the total reward to increment when using feedback 

- `--embedding_model`: BERT model to use when comparing embeddings 

- `--sim_thresh`: Similarity threshold for comparing embedding with centroid

- `--declare_set`: If True, declare all the possiblities in first prompt


## Implement Note

- The root of MISQ (stored in `roots/`) with the same setting will be loaded by default. And broken root file do cause error. Thus, if some errors occur when rerunning an experiment, you can try deleting the related root file.

---
Parts of this code are adopted from the baseline UoT (Hu et al., 2024)
