from base.base_dataset import BaseADDataset
from networks.main import build_network
from optim.cvdd_trainer import CVDDTrainer
import torch
import json


class CVDD(object):
    """A class for Context Vector Data Description (CVDD) models."""

    def __init__(self):
        """Init CVDD instance."""

        # CVDD network: pretrained_model (word embedding or language model) + self-attention module + context vectors
        self.net_name = None
        self.net = None

        self.trainer = None
        self.optimizer_name = None

        self.results = {
            'train_time': None,
        }

    def set_network(self, net_name, dataset, pretrained_model, embedding_size=None, attention_size=150,
                    n_attention_heads=3,clusters=4):
        """Builds the CVDD network composed of a pretrained_model, the self-attention module, and context vectors."""
        self.net_name = net_name
        self.net = build_network(net_name, dataset, embedding_size=embedding_size, pretrained_model=pretrained_model,
                                 update_embedding=False, attention_size=attention_size,
                                 n_attention_heads=n_attention_heads,clusters=clusters)

    def train(self, dataset: BaseADDataset, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 25,
              lr_milestones: tuple = (), lambda_p: float = 1.0,
              alpha: float=1, weight_decay: float = 0.5e-6, device: str = 'cuda',
              n_jobs_dataloader: int = 0):
        """Trains the CVDD model on the training data."""
        self.optimizer_name = optimizer_name
        self.trainer = CVDDTrainer(optimizer_name, lr, n_epochs, lr_milestones, lambda_p, alpha,
                                   weight_decay, device, n_jobs_dataloader)
        self.net = self.trainer.train(dataset, self.net)

        #Get results
        self.results['train_time'] = self.trainer.train_time


    def save_model(self, export_path):
        """Save CVDD model to export_path."""
        torch.save(self.net.state_dict(), export_path)
    

    def load_model(self, import_path, device: str = 'cuda'):
        """Load CVDD model from import_path."""
        self.net.load_state_dict(torch.load(import_path,map_location=device))
        pass

    def save_results(self, export_json):
        """Save results dict to a JSON-file."""
        with open(export_json, 'w') as fp:
            json.dump(self.results, fp)
