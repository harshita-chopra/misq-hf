import os
import json

from misq.chat_utils import import_prompts_by_task
from misq.misq import MISQNode


class SPTask:
    def __init__(self, args):
        self.__dict__.update(vars(args))
        self.free_answer = True  # Explanation is natural language
        self.max_turn = args.max_turn
        self.prompts = import_prompts_by_task("sp")
        self.data = self.load_dataset(args.dataset)
        self.root = None
        self.set = []  # candidate hypotheses
        self.clusters = {}
        self.clusters_text = {}

    def load_dataset(self, name):
        with open(os.path.join(os.path.dirname(__file__), f"../data/{name}.json"), 'r') as file:
            instances = json.load(file)

        return instances

    def create_root(self, instance, root=None):
        self.set = [child["value"] for child in instance["story_tree"]["children"]] if self.open_set_size <= 0 else self.set
        self.repo = instance["self_repo"] 
        self.core = instance["core_sentence"]
        self.key_questions = instance["key_question"]

        if not root:
            self.root = MISQNode("ROOT", True, self.set, None, self.guesser_model)
        else:
            root.set_config(self.n_extend_layers, not self.none_acc_reward, self.expected_reward_method)
            self.root = root
