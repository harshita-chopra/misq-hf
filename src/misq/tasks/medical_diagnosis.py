import os
import json

from misq.chat_utils import import_prompts_by_task
from misq.misq import MISQNode


class MDTask:
    def __init__(self, args):
        self.__dict__.update(vars(args))
        self.free_answer = False
        self.max_turn = 6   # earlier was 5
        self.prompts = import_prompts_by_task("md")
        self.set = []
        self.data = self.load_dataset(args.dataset)
        self.root = None
        self.clusters = {}  # Map of cluster ID -> list of embeddings (for cosine similarity)
        self.clusters_text = {}  # Map of cluster ID -> list of self-reports 

    def load_dataset(self, name):
        if name == "DX":
            self.set = ['Allergic rhinitis', 'upper respiratory tract infection (URTI)', 'pneumonia',
                        'Hand foot and mouth disease', 'Infantile diarrhea']\
                if self.open_set_size <= 0 else self.set
            self.max_turn = 5
        elif name == "MedDG":
            self.free_answer = True
            self.set = ['Enteritis', 'Gastritis', 'Gastroenteritis', 'Esophagitis',
                        'Cholecystitis', 'Appendicitis', 'Pancreatitis', 'Gastric ulcer',
                        'Constipation', 'Cold', 'Irritable bowel syndrome', 'Diarrhea',
                        'Allergic rhinitis', 'Upper respiratory tract infection', 'Pneumonia']\
                if self.open_set_size <= 0 else self.set
            self.max_turn = 6
        elif name == "USMLE":
            # self.free_answer = True
            self.set = ['Atrial septal defect', 'Pulmonary contusion',
                        'Acute pericarditis', 'Sarcoidosis', 'Abdominal aortic aneurysm',
                        'Hypertrophic cardiomyopathy', 'Tetralogy of Fallot',
                        'Ichthyosis vulgaris', 'Vitiligo', 'Seborrheic keratosis',
                        'Pemphigus vulgaris', 'Actinic keratosis',
                        'Squamous cell carcinoma', 'Aromatase deficiency',
                        'Craniopharyngioma', 'Aldosteronoma', 'Papillary carcinoma',
                        'Acute pancreatitis', 'Primary sclerosing cholangitis',
                        'Dubin-Johnson syndrome', 'Crohn disease',
                        'Acalculous cholecystitis', 'Acute mesenteric ischemia',
                        'Acute myelogenous leukemia', 'Intraductal papilloma',
                        'Fibroadenoma', 'Chronic lymphocytic leukemia',
                        'Chronic myeloid leukemia', 'Hemophilia A', 'Pneumonia',
                        'Disseminated gonococcal infection', 'Osteomyelitis',
                        'Infectious mononucleosis', 'Histoplasmosis', 'String test',
                        'Genital herpes', 'Complex partial seizure', 'Meningioma',
                        'Amyotrophic lateral sclerosis', 'Conversion disorder',
                        'Trigeminal neuralgia', 'Placenta previa', 'Pseudocyesis',
                        'Laparoscopy', 'Pelvic inflammatory disease',
                        'Schizoaffective disorder', 'Schizophreniform disorder',
                        'Schizophrenia', 'Cystic fibrosis',
                        'Attention-deficit/hyperactivity disorder',
                        'Autism spectrum disorder', 'Ventricular septal defect', 'Roseola',
                        'Pyelonephritis', 'Poststreptococcal glomerulonephritis',
                        'Membranous nephropathy', 'Osteoarthritis']\
                if self.open_set_size <= 0 else self.set
            self.max_turn = 6
        else:
            raise NotImplementedError
        # return json.loads(os.path.join(os.path.dirname(__file__), f"../data/{name}.json").read())
        with open(os.path.join(os.path.dirname(__file__), f"../data/{name}.json"), 'r') as file:
            return json.load(file)

    def create_root(self, root=None):
        if not root:
            self.root = MISQNode("ROOT", True, self.set, None, self.guesser_model)
        else:
            root.set_config(self.n_extend_layers, not self.none_acc_reward, self.expected_reward_method)
            self.root = root
