import copy, pickle, os
import time

from misq.chat_utils import renew_open_set
from misq.models import get_response_method
from misq.misq import select, renew_node_to_root
from misq.misq import PromptTracker

from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import numpy as np
import torch

class EmbeddingManager:
    def __init__(self, model_name, cache_file, cluster_cache_file, centroid_method='medoid'):
        self.model_name = model_name
        self.cache_file = cache_file
        self.cluster_cache_file = cluster_cache_file
        self.centroid_method = centroid_method
        self.embeddings_cache = {}
        self.clusters_cache = {}

        # Load Hugging Face model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()  # Set model to evaluation mode (no gradient computation)

        self.load_cache()
        self.load_clusters()

        self.runtime_stats = {
            'embedding_time': [],
            'cluster_match_time': [],
            'cluster_assign_time': []
        }
    
    def load_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                self.embeddings_cache = pickle.load(f)
    
    def save_cache(self):
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.embeddings_cache, f)

    def load_clusters(self):
        if os.path.exists(self.cluster_cache_file):
            with open(self.cluster_cache_file, 'rb') as f:
                self.clusters_cache = pickle.load(f)

    def save_clusters(self, clusters):
        self.clusters_cache = clusters
        with open(self.cluster_cache_file, 'wb') as f:
            pickle.dump(self.clusters_cache, f)
    
    def get_clusters(self):
        return self.clusters_cache

    def get_embedding(self, text):
        start = time.time()
        if text in self.embeddings_cache:
            embedding = self.embeddings_cache[text]
            elapsed = time.time() - start
            self.runtime_stats['embedding_time'].append(elapsed)
            return embedding

        # Tokenize input text and create input tensors
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        
        # Compute embeddings using the model
        with torch.no_grad():  # Disable gradient computation for inference
            outputs = self.model(**inputs)
            # Use the [CLS] token embedding (first token) as the sentence embedding
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

        # Normalize embedding
        embedding = normalize([embedding])[0]
        elapsed = time.time() - start
        self.runtime_stats['embedding_time'].append(elapsed)

        # Cache the embedding for reuse
        self.embeddings_cache[text] = embedding
        self.save_cache()
        return embedding
    
    def compute_centroid(self, embeddings):
        if self.centroid_method == 'mean':
            return normalize([np.mean(embeddings, axis=0)])[0]
        elif self.centroid_method == 'medoid':
            pairwise_similarities = cosine_similarity(embeddings)
            medoid_index = np.argmax(pairwise_similarities.sum(axis=1))
            return embeddings[medoid_index]

def assign_to_cluster(embedding, clusters, embedding_manager, threshold=0.85):
    import time
    match_start = time.time()
    if not clusters:
        match_elapsed = time.time() - match_start
        embedding_manager.runtime_stats['cluster_match_time'].append(match_elapsed)
        assign_start = time.time()
        new_cluster_id = 0
        assign_elapsed = time.time() - assign_start
        embedding_manager.runtime_stats['cluster_assign_time'].append(assign_elapsed)
        return new_cluster_id
    max_similarity = -1
    best_cluster_id = None
    for cluster_id, embeddings in clusters.items():
        centroid = embedding_manager.compute_centroid(embeddings)
        similarity = cosine_similarity([embedding], [centroid])[0][0]
        if similarity > max_similarity:
            max_similarity = similarity
            best_cluster_id = cluster_id
    match_elapsed = time.time() - match_start
    embedding_manager.runtime_stats['cluster_match_time'].append(match_elapsed)
    assign_start = time.time()
    if max_similarity >= threshold:
        assign_elapsed = time.time() - assign_start
        embedding_manager.runtime_stats['cluster_assign_time'].append(assign_elapsed)
        return best_cluster_id
    else:
        new_cluster_id = len(clusters)
        assign_elapsed = time.time() - assign_start
        embedding_manager.runtime_stats['cluster_assign_time'].append(assign_elapsed)
        return new_cluster_id
    
def get_examiner_response(task, history):
    response = get_response_method(task.examiner_model)
    msg = [history[0]] + history[-3:] if len(history) > 3 else history
    return response(msg, model=task.examiner_model)


def get_guesser_response(task, history, ques_id, node, cluster_id=None):
    import time
    response = get_response_method(task.guesser_model)

    def simplify_rsp(rsp):
        gpt3_response = get_response_method(task.guesser_model) #  ("gpt-3.5-turbo")
        if len(rsp.split(" ")) > task.expected_action_tokens:
            m = [{"role": "user", "content": task.prompts.extract_q_prompt.format(rsp=rsp)}]
            rsp = gpt3_response(m, model=(task.guesser_model),  # ("gpt-3.5-turbo"),
                                max_tokens=task.expected_action_tokens)
        return rsp

    if len(node.items)==1:
        # leaf node case, only one item -> ask exactly that item 
        target_question = task.prompts.target_question_FA if task.free_answer else task.prompts.target_question
        # but only if it's not already asked
        if target_question.format(target=node.items[0]) not in [h["content"] for h in history] and task.task != 'sp': 
            return node, target_question.format(target=node.items[0]), False
        else: # otherwise send the whole history dialogs and generate new question
            targeting_prompt_free = task.prompts.targeting_prompt_free_FA if task.free_answer else task.prompts.targeting_prompt_free
            msg = copy.deepcopy(history) + [{"role": "user", "content": targeting_prompt_free}]
            res = simplify_rsp(response(msg, model=task.guesser_model)) if task.task != 'sp' else response(msg, model=task.guesser_model)
            return node, res, False
    
    if ques_id < max(4, int(task.max_turn*task.delta)): 
        print("\n\n>>> node_item_set",node.items,"\n")
        select_start = time.time()
        n = select(task, node, cluster_id)
        select_elapsed = time.time() - select_start

        if not hasattr(task, 'runtime_stats'):
            task.runtime_stats = {}
        if 'select_time' not in task.runtime_stats:
            task.runtime_stats['select_time'] = []
        task.runtime_stats['select_time'].append(select_elapsed)
        if n:
            return n, n.question, True

    # towards the end, make new questions 
    if task.task=='sp' and task.open_set_size > 0:
        targeting_prompt_free = task.prompts.targeting_prompt_free_FA if task.free_answer else task.prompts.targeting_prompt_free
        msg = copy.deepcopy(history) + [{"role": "user", "content": targeting_prompt_free}]
        res = simplify_rsp(response(msg, model=task.guesser_model)) if task.task != 'sp' else response(msg, model=task.guesser_model)
        return node, res, False

    # towards the end, make new questions 
    targeting_prompt_set = task.prompts.targeting_prompt_set_FA if task.free_answer else task.prompts.targeting_prompt_set
    
    itemSet = []  # eliminate items already asked about
    questionsAsked = ' '.join([h["content"] for h in history if h["role"]!="user"])
    for item in node.items:
        if item.lower().strip() not in questionsAsked.lower():
            itemSet.append(item.strip())
    if itemSet:
        msg = copy.deepcopy(history) + [{"role": "user", 
                                        "content": targeting_prompt_set.format(item_list_str=', '.join(itemSet),
                                                                                # reminder=' from the initial set of possibilities' if task.declare_set else ''
                                                                                )}]
        print("\n\n>>> targeting_prompt_set",itemSet,"\n")
    else:
        targeting_prompt_free = task.prompts.targeting_prompt_free_FA if task.free_answer else task.prompts.targeting_prompt_free
        msg = copy.deepcopy(history) + [{"role": "user", "content": targeting_prompt_free}]
    res = simplify_rsp(response(msg, model=task.guesser_model)) if task.task != 'sp' else response(msg, model=task.guesser_model)
    return node, res, False


def get_guesser_naive_response(task, history, ques_id):
    response = get_response_method(task.guesser_model)

    msg = copy.deepcopy(history)
    prompt = ""
    if ques_id > int(task.max_turn*0.7):
        prompt += task.prompts.urge_prompt
        if task.inform:
            prompt += task.prompts.inform_prompt.format(item_list_str=', '.join(task.set))
    prompt += "\nYou must reply me with 1 question to ask only."
    msg[-1]["content"] += " " + prompt
    rsp = response(msg, model=task.guesser_model)

    def extract_ques(rsp):
        gpt3_response = get_response_method(task.guesser_model) 
        message = [{"role": "user", "content": task.prompts.extract_q_prompt.format(rsp=rsp)}]
        return gpt3_response(message, model=task.guesser_model) 

    return extract_ques(rsp) if len(rsp.split(" ")) > task.expected_action_tokens else rsp

def extract_conv_hist(sample):
    target = sample["target"]
    path = []

    def dfs(node):
        path.append(node["value"])
        if node["value"] == target:
            return True
        for child in node.get("children", []):
            if dfs(child):
                return True
        path.pop()
        return False

    dfs(sample["story_tree"])
    return "\n".join(path)


def converse(task, i):
    import time
    item = task.data[i]["target"]
    target_decl = task.prompts.target_declaration.format(target=item)
    print("\n\n")
    print(target_decl)
    print("------ DIALOGUE START ------")
    count = 0

    # Reset the prompt call count
    PromptTracker.reset()
    print(f"QGC start: {PromptTracker.get_count()}")

    runtime_stats = {
        'emb_cluster_time': [],
        'select_time': []
    }

    if not task.free_answer:
        history_e = [{'role': 'user', 'content': task.prompts.examiner_prologue.format(item=item)}]
    else:
        if task.task == 'md' and "conv_hist" in task.data[i]:
            history_e = [{'role': 'user', 'content': task.prompts.simulator_prologue.format(item=item, conv_hist=task.data[i]["conv_hist"])}]
        else: # sp
            history_e = [{'role': 'user', 'content': task.prompts.simulator_prologue.format(item=item, conv_hist=extract_conv_hist(task.data[i]))}]

    X ='X' if task.task=='20q' else 'Final outcome'
    closed_set_declaration = f"\n{X} is strictly one of the following: \n{task.set}\n" if (task.declare_set and len(task.set)) else ""
    print(f"\ntask.declare_set = {task.declare_set} \nclosed_set_declaration: {closed_set_declaration}")

    if "self_repo" in task.data[i] and task.open_set_size <= 0:
        print("self_repo in task data")
        guesser_prologue = task.prompts.guesser_prologue_FA if task.free_answer else task.prompts.guesser_prologue
        history_g = [{'role': 'user', 'content': guesser_prologue.format(repo=task.data[i]["self_repo"], 
                                                                         closed_set_declaration=closed_set_declaration)}]
        # print("Self-report:", task.data[i]["self_repo"])
        node = task.root.handle_self_repo(task, task.data[i]["self_repo"])

        if task.feedback_upd:
            # Get embedding using embedding manager
            embedding = task.embedding_manager.get_embedding(task.data[i]["self_repo"])
            
            # Assign to cluster using embedding manager
            cluster_id = assign_to_cluster(embedding, task.clusters, task.embedding_manager, 
                                           threshold=task.sim_thresh)
            if cluster_id not in node.cluster_success_bonus:
                node.cluster_success_bonus[cluster_id] = 0  # Initialize if not present
            print(len(task.clusters), [len(v) for v in task.clusters.values()])
            # Combine all embedding/cluster times into one list
            emb_times = task.embedding_manager.runtime_stats['embedding_time'][:]
            match_times = task.embedding_manager.runtime_stats['cluster_match_time'][:]
            assign_times = task.embedding_manager.runtime_stats['cluster_assign_time'][:]
            runtime_stats['emb_cluster_time'] = emb_times + match_times + assign_times
        else:
            cluster_id = None

    else:
        cluster_id = None
        print("no self_repo in task data")
        history_g = [{'role': 'user', 'content': task.prompts.guesser_prologue.format(repo=task.data[i]["self_repo"], 
                                                                                      closed_set_declaration=closed_set_declaration)}]  

        # !! for openset misq !!
        if task.open_set_size > 0 and task.n_pre_ask > 0:
            print(f"\n\n>>> Open Set, n_pre_ask: {task.n_pre_ask} \n")
            for _ in range(task.n_pre_ask):
                # Question by QG: guesser / question generator / bot1
                bot1_response = get_guesser_naive_response(task, history_g, count+1)  
                print("QG:", bot1_response)   
                history_g.append({'role': 'system', 'content': bot1_response})  # guesser is the system in history_g : to store what questions it genereated previously
                history_e.append({'role': 'user', 'content': bot1_response})    # guesser is the user in history_e   : to store questions it was asked
                # Answer by RE: examiner / response generator / bot2
                bot2_response = get_examiner_response(task, history_e)                
                print("RE:", bot2_response)   
                history_g.append({'role': 'user', 'content': bot2_response})    # examiner is the user in history_g   : to store answers it was given
                history_e.append({'role': 'system', 'content': bot2_response})  # examiner is the system in history_e : to store what answers it genereated previously
                count += 1
                print('------', count, '-------------')
        
        node = task.root.handle_self_repo(task, history_g) if task.open_set_size > 0 else task.root
        if task.open_set_size > 0:
            closed_set_declaration = f"\n{X} is one of the following: \n{node.items}\n" if (task.declare_set and len(node.items)) else ""
            print(f"\nInitialized: task.declare_set = {task.declare_set} \nclosed_set_declaration: {closed_set_declaration}")
            history_g.append({'role': 'user', 'content': closed_set_declaration})

        print("\nin method converse: ", node.print())

    node, bot1_response, flag = get_guesser_response(task, history_g, count + 1, node, cluster_id)
    print("QG:", bot1_response)  # QG: guesser / question generator

    history_g.append({'role': 'system', 'content': bot1_response})  # guesser is the system in history_g
    history_e.append({'role': 'user', 'content': bot1_response})    # guesser is the user in history_e

    # Collect select time if present
    if hasattr(task, 'runtime_stats') and 'select_time' in task.runtime_stats:
        runtime_stats['select_time'] = task.runtime_stats['select_time'][:]

    while True:
        bot2_response = get_examiner_response(task, history_e)  # chatbot 2 is the examiner in code
        update_node = True if bot1_response==node.question else False
        if task.free_answer and flag:
            print("\n\n>>> Free answer mode, handling free answer response")
            node = node.handle_free_answer(task, bot1_response, bot2_response)
        elif bot2_response.replace('\n','').lower().strip().startswith("yes") and update_node:
            node = node.ans2node(True)
        elif bot2_response.replace('\n','').lower().strip().startswith("no") and update_node:
            node = node.ans2node(False)
        history_g.append({'role': 'user', 'content': bot2_response})
        history_e.append({'role': 'system', 'content': bot2_response})
        print("RE:", bot2_response)

        if "guessed it" in bot2_response or "are right." in bot2_response:
            state = 1
            if task.feedback_upd and "self_repo" in task.data[i]:
                print("Propagating feedback...")
                # Update success bonuses along the path to root
                current_node = node
                decay_factor = 0.9  # Decay factor for earlier nodes
                while current_node is not None:
                    bonus = task.bonus_factor * current_node.total_reward * (decay_factor ** current_node.depth)
                    if item in current_node.items:
                        current_node.update_cluster_success_bonus(cluster_id, bonus_increment=bonus)
                    current_node = current_node.parent

                # Successful: Now add embedding to the cluster for usage by new data points
                if cluster_id not in task.clusters:
                    task.clusters[cluster_id] = [embedding]
                else:
                    task.clusters[cluster_id].append(embedding)
                # Save the updated cluster cache
                task.embedding_manager.save_clusters(task.clusters)
            
            break

        count += 1
        print('------', count, '-------------')

        if count >= task.max_turn:
            print("RE: Sorry, time's up. You lose this game.", target_decl)
            state = -1
            break

        # renew
        if count <= int(task.max_turn*0.3) + task.n_pre_ask and task.open_set_size > 0 and len(node.items) < task.size_to_renew:
            print("\nrenew_node_to_root")
            node = renew_node_to_root(task, node, history_g)
            # declare new set if allowed
            closed_set_declaration = f"\nNow {X} is possibly one of the following: \n{node.items}\n" if (task.declare_set and len(node.items)) else ""
            print(f"\nRenewed: task.declare_set = {task.declare_set} \nclosed_set_declaration: {closed_set_declaration}")
            history_g.append({'role': 'user', 'content': closed_set_declaration})

        node, bot1_response, flag = get_guesser_response(task, history_g, count + 1, node, cluster_id)
        print("QG:", bot1_response)
        history_g.append({'role': 'system', 'content': bot1_response})
        history_e.append({'role': 'user', 'content': bot1_response})

    if count < task.max_turn:
        state = 1
    
    print(f"QGC end: {PromptTracker.get_count()}")

    return {'log': {
            'turn': count, 'history_g': history_g, 'history_e': history_e, 'state': state, 'item': task.data[i]["target"],
            'qg_calls': PromptTracker.get_count()
        }, 'runtime_stats': runtime_stats}


def naive_converse(task, i):
    item = task.data[i]["target"]
    target_decl = task.prompts.target_declaration.format(target=item)
    print(target_decl)

    if "self_repo" in task.data[i]:
        guesser_prologue = task.prompts.guesser_prologue_FA if task.free_answer else task.prompts.guesser_prologue
        history_g = [{'role': 'user', 'content': guesser_prologue.format(repo=task.data[i]["self_repo"])}]
        print("Self-report:", task.data[i]["self_repo"])
    else:
        history_g = [{'role': 'user', 'content': task.prompts.guesser_prologue}]

    if not task.free_answer:
        history_e = [{'role': 'user', 'content': task.prompts.examiner_prologue.format(item=item)}]
    else:
        history_e = [{'role': 'user', 'content': task.prompts.simulator_prologue.format(item=item, conv_hist=task.data[i]["conv_hist"])}]

    print("------ DIALOGUE START ------")
    count = 0

    bot1_response = get_guesser_naive_response(task, history_g, count+1)
    print("QG:", bot1_response)

    history_g.append({'role': 'system', 'content': bot1_response})
    history_e.append({'role': 'user', 'content': bot1_response})

    while True:
        bot2_response = get_examiner_response(task, history_e)
        history_g.append({'role': 'user', 'content': bot2_response})
        history_e.append({'role': 'system', 'content': bot2_response})
        print("RE:", bot2_response)

        if "guessed it" in bot2_response or "are right." in bot2_response:
            state = 1
            break

        count += 1
        print('------', count, '-------------')

        if count >= task.max_turn:
            print("Bot 1: Sorry, time's up. You lose this game.", target_decl)
            state = -1
            break

        bot1_response = get_guesser_naive_response(task, history_g, count+1)
        print("QG:", bot1_response)
        history_g.append({'role': 'system', 'content': bot1_response})
        history_e.append({'role': 'user', 'content': bot1_response})

    if count < task.max_turn:
        state = 1

    return {'turn': count, 'history_g': history_g, 'history_e': history_e, 'state': state, 'item': task.data[i]["target"]}
