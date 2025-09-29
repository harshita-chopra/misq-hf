import os
import json
import argparse
import pickle, traceback
import time; trials=3

from tqdm import tqdm

from misq.tasks import get_task
from misq.method import converse, naive_converse, EmbeddingManager
from misq.eval import evaluate_performance


def run(args):
    task = get_task(args)

    args.task_start_index = max(args.task_start_index, 0)
    if args.task_end_index < 0:
        args.task_end_index = len(task.data)
    else:
        args.task_end_index = min(args.task_end_index, len(task.data))

    if args.naive_run:
        log_file = (f'./logs/{args.task}/{args.guesser_model}_as_guesser/{args.desc}_{args.dataset}_{args.temperature}'
                    f'_naive_{"" if args.inform else "un"}inform_EXAMINER{args.examiner_model}'
                    f'_{args.task_start_index}-{args.task_end_index}.json')
    else:
        log_file = (f'./logs/{args.task}/{args.guesser_model}_as_guesser/'
                    f'{f"OS_init{args.open_set_size}_renew{args.size_to_renew}_" if args.open_set_size > 0 else ""}'
                    f'{f"pre{args.n_pre_ask}_" if args.n_pre_ask > 0 else ""}'
                    f'{args.desc}_{args.dataset}_sim{args.mcts_criterion}{"_feedback_upd" if args.feedback_upd else ""}{"_bonus_factor"+str(args.bonus_factor).replace(".","-") if args.feedback_upd else ""}{"_fullset" if args.fullset else ""}{"_deep_simulate" if args.deep_simulate else ""}{"_novisit_reward" if args.novisit_reward else ""}_{"declare_set" if args.declare_set else ""}_{"prompt_w_parents" if args.prompt_w_parents else ""}_{args.temperature}_lambda{args.reward_lambda}_acc{not args.none_acc_reward}'
                    f'_exp{args.expected_reward_method}_L{args.n_extend_layers}_K{args.n_potential_actions}'
                    f'_PRUN{args.n_pruned_nodes}_EXAMINER{args.examiner_model}'
                    f'_{args.task_start_index}-{args.task_end_index}.json')
        root_file = (f'./roots/{args.task}/{args.guesser_model}'
                     f'{f"OS_init{args.open_set_size}_renew{args.size_to_renew}_" if args.open_set_size > 0 else ""}'
                     f'_{args.desc}_{args.dataset}_sim{args.mcts_criterion}{"_feedback_upd" if args.feedback_upd else ""}{"_bonus_factor"+str(args.bonus_factor).replace(".","-") if args.feedback_upd else ""}{"_fullset" if args.fullset else ""}{"_deep_simulate" if args.deep_simulate else ""}{"_novisit_reward" if args.novisit_reward else ""}_{"declare_set" if args.declare_set else ""}_{"prompt_w_parents" if args.prompt_w_parents else ""}_{args.temperature}_root.pickle')
        if os.path.exists(root_file):
            r = open(root_file, 'rb')
            root = pickle.load(r)
            task.create_root(root)
        else:
            os.makedirs(os.path.dirname(root_file), exist_ok=True)
            task.create_root()
            pickle.dump(task.root, open(root_file, 'wb'))
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Initialize embedding manager
    embedding_manager = EmbeddingManager(
        model_name=args.embedding_model,
        cache_file=f'./logs/embs_{args.task}_{args.dataset}.pkl',
        cluster_cache_file=f'./logs/clusters_{args.task}_{args.dataset}{"OS" if args.open_set_size>0 else ""}{args.desc}_sim{args.mcts_criterion}{"_bonus_factor"+str(args.bonus_factor).replace(".","-") if args.feedback_upd else ""}{"_deep_simulate" if args.deep_simulate else ""}_{"declare_set" if args.declare_set else ""}_{"prompt_w_parents" if args.prompt_w_parents else ""}{args.guesser_model}_as_guesser.pkl',
        centroid_method=args.centroid_method,
    )
    # Add embedding manager to task object
    task.embedding_manager = embedding_manager
    # Ensure task clusters are loaded from the manager
    task.clusters = embedding_manager.get_clusters()

    logs = []
    runtime_stats_all = []
    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8') as f:
            logs = json.loads(f.readline())
        args.task_start_index = len(logs)

    error_idx = []
    # loop over main data points
    for i in tqdm(range(args.task_start_index, args.task_end_index)):
        if task.dataset=='MedDG' and task.data[i]['target']=='Cold':
            # skip 'Cold'
            continue
        if args.naive_run:
            log = naive_converse(task, i)
            logs.append(log)
            runtime_stats_all.append({})
        else:
            try:
                result = converse(task, i)
                log = result['log'] if isinstance(result, dict) and 'log' in result else result
                stats = result['runtime_stats'] if isinstance(result, dict) and 'runtime_stats' in result else {}
                logs.append(log)
                runtime_stats_all.append(stats)
            except Exception as e:
                print(f"Error: {e} \n {traceback.print_exc()}")
                error_idx.append(i)
            pickle.dump(task.root, open(root_file, 'wb'))
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps(logs) + '\n')
    # Save runtime stats for all users
    with open(log_file.replace('.json', '_runtime_stats.json'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(runtime_stats_all) + '\n')
    # retry the data points which threw error
    for i in tqdm(error_idx):
        if task.dataset=='MedDG' and task.data[i]['target']=='Cold':
            # skip 'Cold'
            continue
        if args.naive_run:
            log = naive_converse(task, i)
            logs.append(log)
            runtime_stats_all.append({})
        else:
            result = converse(task, i)
            log = result['log'] if isinstance(result, dict) and 'log' in result else result
            stats = result['runtime_stats'] if isinstance(result, dict) and 'runtime_stats' in result else {}
            logs.append(log)
            runtime_stats_all.append(stats)
            pickle.dump(task.root, open(root_file, 'wb'))
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps(logs) + '\n')
    with open(log_file.replace('.json', '_runtime_stats.json'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(runtime_stats_all) + '\n')
    evaluate_performance(log_file, task)


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--guesser_model', type=str, default='gpt-3.5-turbo',
                      choices=['gpt-4o', 'gpt-3.5-turbo',
                               'claude-3-opus-20240229', 'claude-3-sonnet-20240229', 'claude-3.5-haiku',
                               'llama-3-3-70b-instruct', 'mixtral-8x7b-instruct',
                               ])
    args.add_argument('--temperature', type=float, default=0)
    args.add_argument('--examiner_model', type=str, default='gpt-4')
    args.add_argument('--delta', type=float, default=0.6,
                      help="The fraction of max_turn to use for the first 4 questions, e.g. 0.6 means 60% of max_turn")

    # Added new mcts-related arguments
    args.add_argument('--desc', type=str, default='') 
    args.add_argument('--mcts_criterion', type=int, default=-1) 
    args.add_argument('--mcts_c', type=float, default=0) 
    args.add_argument('--mcts_iter', type=int, default=100) 
    args.add_argument('--novisit_reward', action='store_true', help="UCB=reward instead of infinity when visit")
    args.add_argument('--deep_simulate', action='store_true', help="Expand nodes during simulation")
    args.add_argument('--prompt_w_parents', action='store_true', help="Send parent context")
    args.add_argument('--declare_set', action='store_true', help="Send the possible X in first prompt")
    # Added new feedback/embedding-related arguments
    args.add_argument('--feedback_upd', action='store_true', help="Clustering, feedback on success")
    args.add_argument('--bonus_factor', type=float, default=0.50)
    args.add_argument('--embedding_model', type=str, default='medicalai/ClinicalBERT', help='HuggingFace model name for embeddings')
    args.add_argument('--centroid_method', type=str, default='medoid', choices=['mean', 'medoid'], help='Method to compute cluster centroids')
    args.add_argument('--sim_thresh', type=float, default=0.90, help='Similarity threshold for clustering')

    args.add_argument('--fullset', action='store_true', help='Start with full superset at root')

    args.add_argument('--task', type=str, default='20q',
                      choices=['20q', 'md', 'tb', 'sp'])
    args.add_argument('--dataset', type=str, default='common',
                      choices=['bigbench', 'common', 'thing', 'DX', 'USMLE', 'MedDG', 'FloDial'])
    args.add_argument('--task_start_index', type=int, default=-1)
    args.add_argument('--task_end_index', type=int, default=-1)
    args.add_argument('--open_set_size', type=int, default=-1)
    args.add_argument('--size_to_renew', type=int, default=-1)  # only used when open_set_size > 0
    args.add_argument('--n_pre_ask', type=int, default=0)  # only used when open_set_size > 0 and data doesn't contain self-repo

    args.add_argument('--naive_run', action='store_true', default=False)
    args.add_argument('--inform', action='store_true', default=False)  # only used when naive_run

    args.add_argument('--reward_lambda', type=float, default=0.4)
    args.add_argument('--n_extend_layers', type=int, default=3)
    args.add_argument('--n_potential_actions', type=int, default=3)
    args.add_argument('--n_pruned_nodes', type=float, default=0)
    # not prun when = 0
    # exact number when > 0 (e.g. 10: Each layer has a maximum of 10 nodes, M or U, remaining)
    # percentage when < 0 (e.g. -0.5: The remaining 50% of nodes in each layer)

    args.add_argument('--expected_action_tokens', type=int, default=50)
    args.add_argument('--expected_target_tokens', type=int, default=10)

    args.add_argument('--none_acc_reward', action='store_true', default=False)
    args.add_argument('--expected_reward_method', type=str, default='avg', choices=['avg', 'max'])

    args = args.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)
    run(args)
