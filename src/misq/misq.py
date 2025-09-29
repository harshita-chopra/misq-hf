import numpy as np
import math, random

from misq.chat_utils import ques_and_cls_given_items, cls_given_repo, initialize_open_set, renew_open_set

import threading

class PromptTracker:
    prompt_call_count = 0
    _lock = threading.Lock()
    @classmethod
    def increment(cls):
        with cls._lock:
            cls.prompt_call_count += 1
    @classmethod
    def reset(cls):
        with cls._lock:
            cls.prompt_call_count = 0
    @classmethod
    def get_count(cls):
        with cls._lock:
            return cls.prompt_call_count


class MISQNode:

    def __init__(self, question, answer, items, parent: 'MISQNode' = None, model="model", reply=None):
        self.children = []
        self.question = question
        self.answer = answer  # True for "YES" and False for "No"
        self.reply = reply
        self.items = items
        self.parent = parent
        self.depth = self.parent.depth + 1 if self.parent else 0
        self.model = model
        self.n_extend_layers = -1
        self.accumulation = True
        self.expected_method = 'avg'
        self.visits = 0  # Track how many times this node has been visited
        self.total_reward = 0  # Track cumulative reward for backpropagation
        self.cluster_success_bonus = {}  # New: Map cluster IDs to success bonus values
        self.cluster_superset_wts = {}  # New: Map cluster IDs 
        self.other_parent_ques = []
        self.print()

    # Other methods remain unchanged...

    def uct_value(self, exploration_constant, novisit_reward, cluster_id=None):
        """
        Calculate the UCT value for this node.
        """            
        if self.visits == 0:
            if novisit_reward:
                return self.total_reward
            else:
                return float('inf')  # Prioritize unvisited nodes
        exploitation = self.total_reward / self.visits
        exploration = exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)
        # Add cluster success bonus if applicable
        cluster_bonus = self.cluster_success_bonus.get(cluster_id, 0) if cluster_id else 0
        return exploitation + exploration + cluster_bonus

    def best_child(self, exploration_constant, novisit_reward=False, cluster_id=None):
        """
        Select the child with the highest UCT value.
        """
        if not self.children:
            print(f"No children found for node: {self.print()}")
            return None  # Handle case where no children exist
        
        all_children = [child for pair in self.children for child in pair]
        
        if exploration_constant==0:   # directly return reward
            return max(all_children, key=lambda child: child.total_reward + 
                                                        (child.cluster_success_bonus.get(cluster_id, 0) if cluster_id else 0)
                       )
        else:  # return UCT value based max
            return max(all_children, key=lambda child: child.uct_value(exploration_constant, novisit_reward, cluster_id))

    def update_cluster_success_bonus(self, cluster_id, bonus_increment=1.0):
        """Update the success bonus for a specific cluster."""
        if cluster_id not in self.cluster_success_bonus:
            self.cluster_success_bonus[cluster_id] = 0
        self.cluster_success_bonus[cluster_id] += bonus_increment
        # keep greater than zero
        self.cluster_success_bonus[cluster_id] = max(self.cluster_success_bonus[cluster_id], 0)
    
    def update_cluster_superset_wts(self, cluster_id, superset, target):
        """Update the success bonus for a specific cluster."""
        if cluster_id not in self.cluster_superset_wts:
            self.cluster_superset_wts[cluster_id] = {key: 1 for key in superset}
        self.cluster_superset_wts[cluster_id][target] *= 1.1  # Increase weight by 10%
       
    def set_config(self, n_extend_layers: int, none_acc: bool, exp: str):
        self.n_extend_layers = n_extend_layers
        self.accumulation = not none_acc
        self.expected_method = exp

    def _create_children_nodes(self, task, items: list, n, asked_ques: list = None):
        items = list(set(items))
        if self.is_terminal:
            return
        # print(f"=> Creating children nodes for {self.print()}...")

        # Traverse parent nodes to create the context
        if task.prompt_w_parents:
            current_node = self
            if current_node.question not in ["ROOT","self-report","renew"]:
                parent_context = [f"{current_node.question} {('Yes' if current_node.answer else 'No') if not task.free_answer else ''}"]+current_node.other_parent_ques 
            else: 
                parent_context = []
            while current_node.parent:  # Traverse up to the root
                if current_node.parent.question not in ["ROOT","self-report","renew"]:  # Exclude "ROOT" 
                    parent_context.append(f"{current_node.parent.question} {('Yes' if current_node.answer else 'No') if not task.free_answer else ''}")
                current_node = current_node.parent
            parent_context.reverse()  # Reverse to get the correct order from root to current node
        else:
            parent_context = None
        ans = ques_and_cls_given_items(task, items, n, asked_ques, parent_context)
        # --- Compute and store split proportions ---
        if not hasattr(task, 'split_props'):
            task.split_props = []
        for a in ans:
            yes_count = len(a["items_yes"])
            no_count = len(a["items_no"])
            total = yes_count + no_count
            if total > 0:
                prop = yes_count / total
                task.split_props.append(prop)
        # --- End split prop tracking ---
        PromptTracker.increment()
        # print(f"Response: {len(ans)} items created") # {[a['question'] for a in ans]}" )
        if not ans:
            print("\nGot no response")
            return
        self.children.extend(
            (MISQNode(a["question"], True, a["items_yes"], parent=self, model=self.model),
             MISQNode(a["question"], False, a["items_no"], parent=self, model=self.model))
            for a in ans
        )

    def find_children(self, task=None, n=None):
        if self.is_terminal:
            # print(f"no children, node is_terminal", self.items)
            return None
        if task and n and (not self.children or len(self.children) < n):
            # print(f"\nLess or no children found...")
            asked_ques = [ns[0].question for ns in self.children] if self.children else []
            self._create_children_nodes(task, self.items, n - len(asked_ques), asked_ques)
        else:
            pass
            # print("Children already exist, returning children...")
        return self.children

    def find_children_sep(self, task=None, n=None, prun=0):
        _children = self.find_children(task, n)
        return_list = [c[0] for c in _children] + [c[1] for c in _children] if _children else None
        if prun < 0 and return_list:
            return_list = sorted(return_list, key=lambda x: x.idiv_reward, reverse=True)[:int(-prun*len(return_list))]
        return return_list

    def handle_self_repo(self, task, repo, translate=False):
        if task.open_set_size > 0:
            a = initialize_open_set(task, repo)
            node_y = MISQNode("self-report", True, a, parent=self, model=self.model)
            node_n = MISQNode("self-report", False, [], parent=self, model=self.model)
        else:
            a = cls_given_repo(task, self.items, repo, translate, self_repo=True)
            node_y = MISQNode("self-report", True, a["items_yes"], parent=self, model=self.model)
            node_n = MISQNode("self-report", False, a["items_no"], parent=self, model=self.model)

        exist_leaves = []
        for c in self.children:
            exist_leaves.extend([c[0], c[1]])
        
        # Compare based on items (sets)
        node_y_items_set = set(node_y.items)
        node_y_items_len = len(node_y.items)

        for leaf in exist_leaves:
            if node_y_items_set == set(leaf.items):
                # print('Found existing self-repo root with same items.\n')
                return leaf
        if node_y_items_len >= 5:
            valid_leaves = [leaf for leaf in exist_leaves 
                            if node_y_items_set.issubset(set(leaf.items)) 
                            # and abs(len(leaf.items) - len(node_y.items)) <= 10
                            ]
            if valid_leaves:
                closest_leaf = min(valid_leaves, key=lambda leaf: abs(len(leaf.items) - node_y_items_len))
                # print('Found existing self-repo root with closest intersection.\n')
                return closest_leaf
                
        self.children.append((node_y, node_n))
        # print('ADDED node_y \n')
        return node_y
    
    def handle_free_answer(self, task, question, answer):
        """
        Handle free-form answers by classifying items and updating tree structure.
        """
        # Classify parent's items instead of current node's items
        repo = f"Doctor: {question}\nPatient: {answer}\n"
        parent_items = self.parent.items if self.parent else self.items
        a = cls_given_repo(task, parent_items, repo, self_repo=False)

        yes_set = set(a["items_yes"])

        # Check parent's children for matching question and classification
        # matching_sibling = None  
        if self.parent:
            for pair in self.parent.children: # siblings
                for sibling in pair:
                    sibling_items_set = set(sibling.items)
                    # Priority 1: Same question AND same classified set
                    if sibling.question == question and sibling_items_set == yes_set:
                        # print("FOUND exact ques and items: returning current node")
                        return sibling
                    # Priority 2: Same classified set only. Return matching sibling.
                    if sibling_items_set == yes_set:
                        # print("Found exact items: different ques:", sibling.question)
                        sibling.other_parent_ques.append(question)
                        # matching_sibling = sibling
                        return sibling 

        # Fallback: Create a new node at this level
        node_y = MISQNode(question, None, a["items_yes"], parent=self.parent if self.parent else self, model=self.model, reply=answer)
        node_n = MISQNode(question, None, a["items_no"], parent=self.parent if self.parent else self, model=self.model, reply=answer)
        if self.parent:
            self.parent.children.append((node_y, node_n))
            # print("No match found. Adding node_y, node_n as siblings")
        else:
            self.children.append((node_y, node_n))
        return node_y

    def ans2node(self, answer: bool):
        return self if self.answer == answer else next(
            (pair[0] if answer else pair[1] for pair in self.parent.children if self.question == pair[0].question),
            None
        )

    @staticmethod
    def reward_function(x, lamb=0.4):
        return ((-x * np.log2(x) - (1 - x) * np.log2(1 - x)) / (1 + abs(2 * x - 1) / lamb)) if x not in [0, 1] else 0

    def count_M_U(self):
        if not self.parent:
            return None
        c_1 = len(self.items)
        c_2 = len(self.ans2node(not self.answer).items)
        return c_1, c_2

    @property
    def idiv_reward(self):
        c = self.count_M_U()
        if not c or abs(c[0] - c[1]) == 1:
            return 1.
        node_reward = self.reward_function(c[0] / (c[0] + c[1]))
        return node_reward

    @staticmethod
    def accumulated_reward(node, level, accum=True):
        term = 0 if (level == 1 or not accum) else node.accumulated_reward(node.parent, level - 1, accum)
        return node.idiv_reward + term

    @staticmethod
    def avg_expected(setting_node, child_list, n_extend_layers, level, prob):
        if not child_list:
            return 0
        child_r = 0.
        for child_node in child_list:
            child_node.set_config(setting_node.n_extend_layers, setting_node.accumulation, setting_node.expected_method)
            child_r += child_node.expected_reward(n_extend_layers, level=level + 1)
        return child_r * prob / len(child_list) if len(child_list) > 0 else 1

    @staticmethod
    def max_expected(setting_node, child_list, n_extend_layers, level, prob):
        if not child_list:
            return 0
        child_r = 0.
        for child_node in child_list:
            child_node.set_config(setting_node.n_extend_layers, setting_node.accumulation, setting_node.expected_method)
            child_r = max(child_node.expected_reward(n_extend_layers, level=level + 1), child_r)
        return child_r * prob

    def expected_reward(self, n_extend_layers, level=1):
        if not self.parent:
            return 1.
        c_1, c_2 = self.count_M_U()
        p = c_1 / (c_1 + c_2)
        partner = self.ans2node(not self.answer)
        if level == self.n_extend_layers - 1 or self.is_terminal or not self.children:
            return self.accumulated_reward(self, level, self.accumulation)
        else:
            expected_function = self.avg_expected if self.expected_method == 'avg' else self.max_expected
            avg_1 = expected_function(self, self.find_children_sep(), n_extend_layers, level, p)
            avg_2 = expected_function(self, partner.find_children_sep(), n_extend_layers, level, 1 - p)
        return (p * (self.idiv_reward + avg_1) +
                (1 - p) * (partner.idiv_reward + avg_2))

    @property
    def reward(self):
        self.set_config(self.parent.n_extend_layers, self.parent.accumulation, self.expected_method)
        return self.expected_reward(self.n_extend_layers)

    @property
    def is_terminal(self):
        return len(self.items) <= 2

    def print(self):
        return f"""question: {self.question}; answer: {self.answer}; items: {len(self.items)}; depth: {self.depth}; visits: {self.visits}; is_terminal: {self.is_terminal}"""

    def __eq__(self, other):
        if isinstance(other, MISQNode):
            if len(self.items) != len(other.items) or self.depth != other.depth:
                return False
            for i in self.items:
                if i not in other.items:
                    return False
            return True
        else:
            return False

    def __lt__(self, other):
        if isinstance(other, MISQNode):
            return self.reward < other.reward
        raise ValueError("Comparison with non-MISQNode object")

    def __gt__(self, other):
        if isinstance(other, MISQNode):
            return self.reward > other.reward
        raise ValueError("Comparison with non-MISQNode object")
    
    # def print_tree(self, level=0):
    #     """
    #     Recursively prints the tree structure starting from the current node.
    #     """
    #     indent = "   " * level  # Indentation for readability
    #     print(f"{indent}- Question: {self.question}, Answer: {self.answer}, Items: {len(self.items)}, Depth: {self.depth}, Visits: {self.visits}, Terminal: {self.is_terminal}")
    #     for child_pair in self.children:
    #         for child in child_pair:
    #             child.print_tree(level + 1)

    def print_tree(self, level=0, start_from_root=True):
        """
        Prints the tree structure starting from the root node or current node.
        If `start_from_root` is True, the function finds the root node and starts printing from there.
        """
        if start_from_root and hasattr(self, 'parent'):  # Ensure the node can navigate upwards
            root = self
            while root.parent is not None:  # Traverse up to find the root
                root = root.parent
            root.print_tree(level=0, start_from_root=False)
            return

        # Print the current node details
        indent = "   " * level  # Indentation for readability
        print(f"{indent}- Question: {self.question}, Answer: {self.answer}, Items: {len(self.items)}, Depth: {self.depth}, Visits: {self.visits}, Terminal: {self.is_terminal}")
        
        # Recursively print children
        for child_pair in self.children:
            for child in child_pair:
                child.print_tree(level + 1, start_from_root=False)


def renew_node_to_root(task, node, history):
    a = renew_open_set(task, history, node.items)
    node_y = MISQNode("renew", True, a, parent=task.root, model=node.model)
    node_n = MISQNode("renew", False, [], parent=task.root, model=node.model)
    exist_leaves = []
    for c in task.root.children:
        exist_leaves.extend([c[0], c[1]])
    if node_y in exist_leaves:
        return exist_leaves[exist_leaves.index(node_y)]
    task.root.children.append((node_y, node_n))
    return node_y


##### MCTS Functions ######


def select(task, root, cluster_id):
    """
    Perform MCTS to select the best question.
    """
    if root.is_terminal:
        return None
    
    # print("\nmcts select root:", root.print())
    iterations = task.mcts_iter  # Number of MCTS iterations (can be tuned)
    
    for _ in range(iterations):
        # Step 1: Selection - Traverse tree using UCT until a leaf is reached
        selected_node = mcts_select(root, exploration_constant=task.mcts_c, novisit_reward=task.novisit_reward, cluster_id=cluster_id)

        # Step 2: Expansion - Expand the selected leaf node if possible (one level)
        if selected_node and not selected_node.is_terminal:
            mcts_expand(task, selected_node)

        # Step 3: Simulation - Simulate a rollout from the expanded node to estimate reward
        simulated_reward = mcts_simulate(selected_node, task.mcts_criterion, task) if selected_node else 0

        # Step 4: Backpropagation - Propagate the simulated reward up the tree
        if selected_node:
            mcts_backpropagate(selected_node, simulated_reward)

    # After all iterations, choose the best child of the root based on visit count or total reward
    best_child = root.best_child(exploration_constant=0)  # Set exploration constant to zero for final selection
   
    return best_child  


def mcts_select(node, exploration_constant=1.41, novisit_reward=False, cluster_id=None):
    """
    Traverse the tree using UCT until a leaf node is reached.
    """
    current_node = node
    while current_node.children:
         # Select between Yes/No branches based on UCT
        best_child = current_node.best_child(exploration_constant, novisit_reward, cluster_id)
        if not best_child:
            # Stop traversal if no children are available
            break  
        current_node = best_child

    return current_node


def mcts_expand(task, node):
    """
    Expand the selected node by creating its children.
    """
    if not node.is_terminal and not node.children:
        node.find_children(task, task.n_potential_actions)  # Create children nodes


def mcts_simulate(node, criterion, task):
    """
    Simulate a rollout from the given node to estimate its reward.
    Use a lightweight heuristic or random sampling to simulate outcomes.
    """
    current_node = node
    depth_limit = 3  # Limit simulation depth to avoid excessive computation
    depth = 0

    while not current_node.is_terminal and depth < depth_limit:
        if not current_node.children:
            if not task.deep_simulate:
                # Stop if no children are available
                break  
            else: # Expand one level
                mcts_expand(task, current_node)
        
        if criterion==1:
            # 1. Randomly select a child for simulation (diverse exploration)
            possible_children = [child for pair in current_node.children for child in pair]
            current_node = random.choice(possible_children)

        depth += 1

    return current_node.reward


def mcts_backpropagate(node, reward):
    """
    Backpropagate the simulated reward up the tree.
    """
    current_node = node
    while current_node is not None:
        current_node.visits += 1
        current_node.total_reward += reward
        current_node = current_node.parent

def softmax(values):
    exp_values = [math.exp(v) for v in values]
    total = sum(exp_values)
    return [v / total for v in exp_values]