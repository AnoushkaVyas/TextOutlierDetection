from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset
from networks.cvdd_Net import CVDDNet
from sklearn.metrics import roc_auc_score

import logging
import time
import torch
import torch.optim as optim
import numpy as np


class CVDDTrainer(BaseTrainer):

    def __init__(self, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150, lr_milestones: tuple = (),
                 lambda_p: float = 0.0, alpha: float=1,
                 weight_decay: float = 1e-6, device: str = 'cuda', n_jobs_dataloader: int = 0):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, weight_decay, device,
                         n_jobs_dataloader)

        self.lambda_p = lambda_p
        self.alpha=alpha


    def project_lambda(self,ll):
        ll=np.where(ll < 0.5 ,0,1)
        return ll

    def compute_loss(self,betak,lambdai,membership,concat_M):

        l1= lambdai.T @ torch.diagonal (concat_M @ concat_M.T)
        l2= torch.sum(torch.mul(membership @ membership.T,torch.mul( concat_M @ concat_M.T, lambdai @ lambdai.T)))
        l3= (betak.T @ ((membership.T @ lambdai)-1)**2).squeeze()

        return l1-l2,l3

    def train(self, dataset: BaseADDataset, net: CVDDNet):
        logger = logging.getLogger()

        # Set device for network
        net = net.to(self.device)

        # Get number of clusters
        clusters=net.clusters

        # Get number of attention heads
        n_attention_heads = net.n_attention_heads

        # Get train data loader
        train_loader, _ = dataset.loaders(batch_size=7769, num_workers=self.n_jobs_dataloader)

        # Define outlier vector
        lambdai=torch.nn.Parameter(torch.rand(7769,1)*self.alpha).to(self.device)

        # Define cluster weight vector
        betak=torch.rand(clusters,1)

        # Set parameters and optimizer (Adam optimizer for now)
        parameters = filter(lambda p: p.requires_grad, net.parameters())
        optimizer = optim.Adam(parameters, lr=self.lr, weight_decay=self.weight_decay)

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Training
        logger.info('Starting training...')
        start_time = time.time()
        net.train()

        lambdas=[]
        labels=[]

        for epoch in range(self.n_epochs):

            scheduler.step()
            if epoch in self.lr_milestones:
                logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            epoch_loss = 0.0
            n_batches = 0
            epoch_start_time = time.time()

            for data in train_loader:
                _, text_batch, label_batch, _ = data
                text_batch = text_batch.to(self.device)
                # text_batch.shape = (sentence_length, batch_size)

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize

                # forward pass
                membership, concat_M, A = net(text_batch)

                # membership,shape=(batch_size, clusters)
                # A.shape = (batch_size, n_attention_heads, sentence_length)
                # concat_M.shape = (batch_size, n_attention_heads*hidden embedding)

                # get orthogonality penalty: P = (AAT - I)
                I = torch.eye(n_attention_heads).to(self.device)
                AAT = A @ A.transpose(1, 2)
                P = torch.mean((AAT.squeeze() - I) ** 2)

                # compute loss
                loss_P = self.lambda_p * P
                loss_1_2,loss_3 = self.compute_loss(betak,lambdai,membership,concat_M)
                lossmax = loss_3-loss_P-loss_1_2

                lossmax.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)  # clip gradient norms in [-0.5, 0.5]
                optimizer.step()

                  # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize

                # forward pass
                membership, concat_M, A = net(text_batch)

                # membership,shape=(batch_size, clusters)
                # A.shape = (batch_size, n_attention_heads, sentence_length)
                # concat_M.shape = (batch_size, n_attention_heads*hidden embedding)

                # get orthogonality penalty: P = (AAT - I)
                I = torch.eye(n_attention_heads).to(self.device)
                AAT = A @ A.transpose(1, 2)
                P = torch.mean((AAT.squeeze() - I) ** 2)

                # compute loss
                loss_P = self.lambda_p * P
                loss_1_2,loss_3 = self.compute_loss(betak,lambdai,membership,concat_M)
                lossmin= loss_P+loss_1_2+loss_3

                lossmin.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)  # clip gradient norms in [-0.5, 0.5]
                optimizer.step()

                epoch_loss += lossmax.item()+lossmin.item()
                n_batches += 1
            
            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            logger.info(f'| Epoch: {epoch + 1:03}/{self.n_epochs:03} | Train Time: {epoch_train_time:.3f}s '
                        f'| Train Loss: {epoch_loss / n_batches:.6f} |')


        self.train_time = time.time() - start_time

        # Log results
        logger.info('Training Time: {:.3f}s'.format(self.train_time))
        logger.info('Finished training.')

        # AUC
        print(roc_auc_score(np.array(labels).flatten(),np.array(lambdas).flatten()))

        return net

