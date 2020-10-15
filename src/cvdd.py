from base.base_dataset import BaseADDataset
from networks.main import build_network
from optim.cvdd_trainer import CVDDTrainer
import torch

class CVDD(object):
    """A class for Context Vector Data Description (CVDD) models."""

    def __init__(self):
        """Init CVDD instance."""

        # CVDD network: pretrained_model (word embedding or language model) + self-attention module + context vectors
        self.net_name = None
        self.net = None

        self.trainer = None

    def set_network(self, net_name, dataset, pretrained_model, embedding_size=None, attention_size=150,
                    n_attention_heads=3,clusters=4):
        """Builds the CVDD network composed of a pretrained_model, the self-attention module, and context vectors."""
        self.net_name = net_name
        self.net = build_network(net_name, dataset, embedding_size=embedding_size, pretrained_model=pretrained_model,
                                 update_embedding=True, attention_size=attention_size,
                                 n_attention_heads=n_attention_heads,clusters=clusters)

    def train(self, dataset: BaseADDataset, device: str = 'cuda',n_jobs_dataloader: int = 0):

        """Trains the CVDD model on the training data."""
        self.trainer = CVDDTrainer (device, n_jobs_dataloader)
        self.net = self.trainer.train(dataset, self.net)

    def save_model(self, export_path):
        """Save CVDD model to export_path."""
        torch.save(self.net.state_dict(), export_path)
        

    def load_model(self, import_path, device: str = 'cuda'):
        """Load CVDD model from import_path."""
        self.net.load_state_dict(torch.load(import_path,map_location=device))
        
