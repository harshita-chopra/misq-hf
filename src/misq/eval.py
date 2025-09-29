import json
import os


def evaluate_performance(file, task):
    cnt = success = 0
    length = success_length = 0
    qg_calls = []
    with open(file, 'r') as f:
        data = json.load(f)
    # Try to load runtime stats if available
    runtime_stats_file = file.replace('.json', '_runtime_stats.json')
    runtime_stats = []
    if os.path.exists(runtime_stats_file):
        with open(runtime_stats_file, 'r') as f:
            runtime_stats = json.load(f)
    emb_cluster_times = []
    select_times = []
    split_props = []
    # Try to load split_props if present in task
    if hasattr(task, 'split_props') and task.split_props:
        split_props = task.split_props
    # Or try to load from a file if you want to persist it
    for i, entry in enumerate(data):
        if task.dataset=='MedDG' and entry['item']=='Cold':
            # skip
            continue
        else:
            cnt += 1
            # Add the actual frequency of QG calls for this dialogue
            qg_calls.append(entry['qg_calls'])
            if entry['state'] == 1:
                success += 1
                success_length += entry['turn']
                length += entry['turn']
            else:
                length += task.max_turn
            # Collect runtime stats if available
            if i < len(runtime_stats):
                stats = runtime_stats[i]
                if 'emb_cluster_time' in stats and stats['emb_cluster_time']:
                    emb_cluster_times.extend(stats['emb_cluster_time'])
                if 'select_time' in stats and stats['select_time']:
                    select_times.extend(stats['select_time'])
            # Collect split_props if present in entry (for future-proofing)
            if 'split_props' in entry:
                split_props.extend(entry['split_props'])
    print('Dialogue Count:', cnt)
    print('Success Rate:', success / cnt)
    print('Mean Conversation Length in Successful Cases:', success_length / success)
    print('Mean Conversation Length:', length / cnt)
    print('Mean QG Calls per Dialogue:', sum(qg_calls) / cnt)
    if emb_cluster_times:
        print('Mean Embedding+Cluster Time per Call:', sum(emb_cluster_times) / len(emb_cluster_times))
    if select_times:
        print('Mean Select Time per Call:', sum(select_times) / len(select_times))
    if split_props:
        print('Mean Split Proportion (yes/(yes+no)):', sum(split_props) / len(split_props))