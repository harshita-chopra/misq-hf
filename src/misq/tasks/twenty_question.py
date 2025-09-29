from misq.chat_utils import import_prompts_by_task
from misq.misq import MISQNode


class Q20Task:
    def __init__(self, args):
        self.__dict__.update(vars(args))
        self.free_answer = False
        self.max_turn = 20
        self.prompts = import_prompts_by_task("20q")
        self.set = []
        self.data = self.load_dataset(args.dataset)
        self.root = None
        self.clusters = {}  # Map of cluster ID -> list of embeddings (for cosine similarity)

    def load_dataset(self, name):
        from misq.data.data_20q import BIG_BENCH_CONCEPT, COMMON, THING200
        if name == "bigbench":
            self.set = BIG_BENCH_CONCEPT if self.open_set_size <= 0 else self.set
            return [{"target": x} for x in BIG_BENCH_CONCEPT]
        elif name == "common":
            self.set = COMMON if self.open_set_size <= 0 else self.set
            return [{"target": x} for x in COMMON]
        elif name == "thing":
            self.set = THING200 if self.open_set_size <= 0 else self.set
            return [{"target": x} for x in THING200]
        else:
            raise NotImplementedError

    def create_root(self, root=None):
        if not root:
            self.root = MISQNode("ROOT", True, self.set, None, self.guesser_model)
        else:
            root.set_config(self.n_extend_layers, not self.none_acc_reward, self.expected_reward_method)
            self.root = root

