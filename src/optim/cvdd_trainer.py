from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset
from networks.cvdd_Net import CVDDNet
from networks.Lambda_Net import LambdaNet
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
from utils.vocab import Vocab
import matplotlib.pyplot as plt

import logging
import time
import torch
import torch.optim as optim
import numpy as np


class CVDDTrainer(BaseTrainer):

    def __init__(self,device: str = 'cuda', n_jobs_dataloader: int = 0):
        super().__init__( device, n_jobs_dataloader)

        self.lambda_p = 0.1
        self.alpha=1.0
        self.n_epochs=5
        self.max_epoch=2
        self.min_epoch=2
        self.lr_max= 0.001
        self.lr_min= 0.001
        self.lr_milestones_max= [2]
        self.lr_milestones_min= [2]
        self.weight_decay_max= 0.5e-4
        self.weight_decay_min= 0.5e-4
        self.gamma_max= 0.1
        self.gamma_min= 0.1
        self.save_results='../log/test_reuters/'

    def compute_loss(self,lambdai,membership,concat_M):

        l1= lambdai.T @ torch.diagonal (concat_M @ concat_M.T)
        l2= torch.sum(torch.mul(membership @ membership.T,torch.mul( concat_M @ concat_M.T, lambdai @ lambdai.T)))
        l3= torch.abs((membership.T @ lambdai)-1)

        return l1-l2,l3

    def train(self, dataset: BaseADDataset, net: CVDDNet):
        logger = logging.getLogger()

        # Set device for network
        net = net.to(self.device)

        # Get number of attention heads
        n_attention_heads = net.n_attention_heads

        # Get embedding size
        hidden_size= net.hidden_size

        # Set device for network
        net_lambda = LambdaNet(n_attention_heads,hidden_size).to(self.device)

        # Get cluster size
        clusters =net.clusters

        # Define cluster weight vector
        betak=torch.ones(clusters,1)
        beta_update = torch.ones(clusters,1)

        # Get train data loader
        data_loader= dataset.loaders(num_workers=self.n_jobs_dataloader)

        # Set parameters and optimizer (Adam optimizer for now)
        parameters_min = filter(lambda p: p.requires_grad, net.parameters())
        optimizer_min = optim.Adam(parameters_min, lr=self.lr_min, weight_decay=self.weight_decay_min)

        parameters_max = filter(lambda p: p.requires_grad, net_lambda.parameters())
        optimizer_max = optim.Adam(parameters_max, lr=self.lr_max, weight_decay=self.weight_decay_max)

        # Set learning rate scheduler
        scheduler_max = optim.lr_scheduler.MultiStepLR(optimizer_max, milestones= self.lr_milestones_max, gamma= self.gamma_max)
        scheduler_min = optim.lr_scheduler.MultiStepLR(optimizer_min, milestones= self.lr_milestones_min, gamma=self.gamma_min)

       # Training
        logger.info('Starting training...')
        start_time = time.time()
        net.train()
        net_lambda.train()

        # losses
        max_loss=[]
        min_loss=[]

        for epoch in range(self.n_epochs):

            scheduler_max.step()
            scheduler_min.step()

            if epoch in self.lr_milestones_max:
                logger.info('  LR scheduler for maximization: new learning rate is %g' % float(scheduler_max.get_lr()[0]))
            
            if epoch in self.lr_milestones_min:
                logger.info('  LR scheduler for minimization: new learning rate is %g' % float(scheduler_min.get_lr()[0]))

            epoch_start_time = time.time()

            for data in data_loader:
                _, text_batch, label_batch, classlabel_batch,_ = data
                text_batch = text_batch.to(self.device)

                for minepoch in range(self.min_epoch):

                    #Zero the network parameter gradients
                    optimizer_min.zero_grad()

                    # Update network parameters via backpropagation: forward + backward + optimize

                    # forward pass
                    membership, concat_M, A = net(text_batch)
                    lambdai=net_lambda(concat_M)

                    # Project lambda
                    with torch.no_grad():
                        lambdai.clamp_(0, self.alpha)

                    # get orthogonality penalty: P = (AAT - I)
                    I = torch.eye(n_attention_heads).to(self.device)
                    AAT = A @ A.transpose(1, 2)
                    P = torch.mean((AAT.squeeze() - I) ** 2)

                    # compute loss
                    loss_P = self.lambda_p * P
                    loss_1_2,loss_3 = self.compute_loss(lambdai,membership,concat_M)

                    flag=0
                    for l in range(clusters):
                        if minepoch>0 and beta_update[l]==1:
                            if  loss_3[l] >= 1.2*old_loss_3[l]:
                                flag=1
                                beta_update[l] = 0
                                betak[l] /= 2
                                print('Penalty Error increases significantly iteration %s, So stopped increasing beta and did the roll back' % (epoch+1))

                    if flag==1:
                        continue

                    old_loss_3=loss_3
                    loss_3= (betak.T @ loss_3).squeeze()
                    lossmin= loss_P+loss_1_2+loss_3

                    lossmin.backward()
                    optimizer_min.step()

                    for l in range(clusters):
                        if beta_update[l]==1:
                            print('Iteration %s, increasing beta for minimization' % (epoch+1))
                            betak[l] *= 2

                    min_loss.append(loss_1_2)

                for maxepoch in range(self.max_epoch):

                    # Zero the network parameter gradients
                    optimizer_max.zero_grad()

                    # Update network parameters via backpropagation: forward + backward + optimize

                    # forward pass
                    membership, concat_M, A = net(text_batch)
                    lambdai=net_lambda(concat_M)

                    # Project lambda
                    with torch.no_grad():
                        lambdai.clamp_(0, self.alpha)

                    # compute loss
                    loss_1_2,loss_3 = self.compute_loss(lambdai,membership,concat_M)

                    flag=0
                    for l in range(clusters):
                        if maxepoch>0 and beta_update[l]==1:
                            if  loss_3[l] >= 1.2*old_loss_3[l]:
                                flag=1
                                beta_update[l] = 0
                                betak[l] /= 2
                                print('Penalty Error increases significantly iteration %s, So stopped increasing beta and did the roll back' % (epoch+1))

                    if flag==1:
                        continue

                    old_loss_3=loss_3
                    loss_3= (betak.T @ loss_3).squeeze()
                    lossmax = loss_3-loss_1_2

                    lossmax.backward()
                    optimizer_max.step()

                    for l in range(clusters):
                        if beta_update[l]==1:
                            print('Iteration %s, increasing beta for maximization' % (epoch+1))
                            betak[l] *= 2

                    max_loss.append(loss_1_2)

                # Save embeddings
                if epoch ==0:
                    filename= self.save_results+'start_M.txt'
                    np.savetxt(filename, concat_M.cpu().data.numpy())

                if epoch == int(self.n_epochs/2):
                    filename= self.save_results+'mid_M.txt'
                    np.savetxt(filename, concat_M.cpu().data.numpy())

                if epoch == self.n_epochs-1:
                    filename=self.save_results+'end_M.txt'
                    np.savetxt(filename, concat_M.cpu().data.numpy())
        
            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            logger.info(f'| Epoch: {epoch + 1:03}/{self.n_epochs:03} | Train Time: {epoch_train_time:.3f}s ')


        self.train_time = time.time() - start_time

        # Log results
        logger.info('Training Time: {:.3f}s'.format(self.train_time))
        logger.info('Finished training.')

        #Saving results
        filename=self.save_results+'outlier.txt'
        np.savetxt(filename, lambdai.cpu().data.numpy())

        filename= self.save_results+"label.txt"
        np.savetxt(filename, label_batch.cpu().data.numpy())

        filename= self.save_results+"classlabel.txt"
        np.savetxt(filename, classlabel_batch.cpu().data.numpy())

        filename= self.save_results+"membership.txt"
        np.savetxt(filename, membership.cpu().data.numpy())

        filename= self.save_results+"minloss.txt"
        np.savetxt(filename, np.array(min_loss))

        filename= self.save_results+"maxloss.txt"
        np.savetxt(filename, np.array(max_loss))


        # Plotting Loss
        plt.plot(max_loss,label='MaxLoss')
        plt.plot(min_loss,label='MinLoss')
        plt.legend()
        plt.savefig(self.save_results+'loss.png')

        return net
