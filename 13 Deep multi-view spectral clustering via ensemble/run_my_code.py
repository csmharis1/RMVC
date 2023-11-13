# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 14:21:50 2023

@author: Chris
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from W_Construct import norm_W, KNN
from data_loader import load_mat
import numpy as np
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, f1_score, adjusted_rand_score
from layers import GraphConvolution
import os
from sklearn.cluster import k_means
import warnings
import argparse
import csv

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='DMCE')
parser.add_argument('--epochs', '-te', type=int, default=84, help='number of train_epochs')
parser.add_argument('--lr', type=float, default=0.0001, help='Adam learning rate')
parser.add_argument('--r', type=float, default=1, help='Scalar to control the distribution of the weights')
parser.add_argument('--dataset_folder', type=str, default='data', help='Folder containing datasets')
parser.add_argument('--k', type=int, default=3, help='The number of neighbors in initialized similarity graphs')
args = parser.parse_args()

class GCN(nn.Module):
    def __init__(self, nfeat,nhid):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        return x

class GRGNN(nn.Module):
    def __init__(self, n_cluster):
        super(GRGNN, self).__init__()
        self.l1 = nn.Linear(n_cluster*n_view*2,n_cluster*n_view)
        self.l2 = nn.Linear(n_cluster*n_view, n_cluster)
        self.gcn = GCN(n_cluster*n_view,n_cluster*n_view*2)
        self.weight = nn.Parameter(torch.full((n_view,), 1.0), requires_grad=True)

    def forward(self,W,LM1):
        weight = F.softmax(self.weight)
        weight = torch.pow(weight,args.r)
        LM = torch.zeros((N, n_cluster, n_view)).to(device)
        for i in range(n_view):
            LM[:,:,i] = LM1[:,:,i]*weight[i]
        LM = LM.split(1,dim=2)
        r_fusion = torch.cat(LM,dim=1).squeeze()
        S_fusion = r_fusion@r_fusion.t()
        W = W.split(1,dim=2)
        F_1 = torch.cat(W,dim=1).squeeze()
        F_gcn = self.gcn(F_1,S_fusion)
        F_gcn = F.selu(self.l1(F_gcn))
        F_r = F.softmax(self.l2(F_gcn), dim=1)
        return F_r, F_1,S_fusion

    def run(self,W,LM,l_1,l_2):
        optimizer = torch.optim.Adam(self.parameters(), lr=args.lr)
        self.to(device)
        for it in range(args.epochs):
            for i in range(100):
                optimizer.zero_grad()
                CF,r_fusion,S = self(W,LM)
                E = CF@CF.t()
                loss_1 = GCLoss(E,S)
                W_new = torch.abs(r_fusion@r_fusion.t())
                loss_or = 4 /(n_cluster * (n_cluster - 1))* triu(torch.t(CF) @ CF)
                loss_gr =torch.pow(torch.norm(E - W_new),2)
                loss = loss_gr+loss_or*l_1+loss_1*l_2
                loss.backward(retain_graph=True)
                optimizer.step()
            y_pred = np.argmax(CF.cpu().detach().numpy(), axis=1)
            ACC = acc(GT, y_pred+1)
            NMI = metrics.normalized_mutual_info_score(GT, y_pred)
            Purity = purity_score(GT, y_pred)
            print('epoch: {},clustering accuracy: {}, NMI: {}, Purity: {}'.format(it, ACC, NMI, Purity))

        return E,ACC,NMI,Purity,S,CF,r_fusion

def acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max())+1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment as linear_assignment
    r_ind,c_ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(r_ind,c_ind)]) * 1.0 / y_pred.size

def purity_score(y_true, y_pred):
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true==labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)
    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred==cluster], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred==cluster] = winner
    return metrics.accuracy_score(y_true, y_voted_labels)
def GCLoss(logits,Lc):

    mask = torch.eye(N, dtype=torch.float32).to(device)
    exp_logits = torch.exp(logits)
    ones_matrix = torch.ones_like(mask)
    ones_matrix[Lc == 0] = 0
    prob = (ones_matrix * exp_logits).sum() / ((torch.ones_like(mask)-ones_matrix) * exp_logits).sum()
    loss = -torch.log(prob).mean()

    return loss

def triu(X):
    return torch.sum(torch.triu(X, diagonal=1))

def label2matrix(label):
    label = np.array(label)
    uq_la = np.unique(label)
    c = uq_la.shape[0]
    n = label.shape[0]
    label_mat = np.zeros((n,c))
    for i in range(c):
        index = (label == i+1)
        label_mat[index,i]=1
    return label_mat


if __name__ == "__main__":
    datasets = [f for f in os.listdir(args.dataset_folder) if os.path.isfile(os.path.join(args.dataset_folder, f))] 
    print(datasets)
    for dataset in datasets:
        print(dataset)
        X, GT = load_mat(os.path.join(args.dataset_folder, dataset))
        n_cluster = len(np.unique(GT))
        N, _ = X[0].shape
        GT = GT.reshape(np.max(GT.shape), )
        n_view = len(X)
        W = torch.zeros((N, N, n_view))
        F0 = torch.zeros((N,n_cluster,n_view))
        LM = torch.zeros((N, n_cluster, n_view))
        lambda_1 = [1e-2,1e-1,1,10,1e2,1e3]
        lambda_2 = [1e-2,1e-1,1,10,1e2,1e3]
        
        
        
        all_results = []
        for k in range(6):
            for l in range(6):
                for i in range(n_view):
                    
                    A = KNN(X[i], args.k)
                    W[:, :, i] = torch.tensor(norm_W(A), dtype=torch.float32)
                    val, vec = np.linalg.eigh(norm_W(A))
                    F1 = vec[:, -n_cluster:]
                    _, labels, _ = k_means(F1, n_clusters=n_cluster)
                    label = label2matrix(labels)
                    LM[:, :, i] = torch.tensor(label, dtype=torch.float32)
                    F0[:, :, i] = torch.tensor(F1, dtype=torch.float32)
                    print("View: {} generated".format(i))
                ACC = []
                NMI = []
                P = []
    
                
                model = GRGNN(n_cluster)
                E, ACC, NMI, Purity, S, CF, r_fusion = model.run(F0.to(device), LM.to(device), lambda_1[k], lambda_2[l])
    
                y_pred = np.argmax(CF.cpu().detach().numpy(), axis=1)
                precision = precision_score(GT, y_pred, average='macro')
                recall = recall_score(GT, y_pred, average='macro')
                f1 = f1_score(GT, y_pred, average='macro')
                rand_index = adjusted_rand_score(GT, y_pred)
    
                results = [k, l, ACC, NMI, Purity, precision, recall, f1, rand_index]
                all_results.append(results)
        with open(f'results_{dataset}.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Lambda1', 'Lambda2', 'Accuracy', 'NMI', 'Purity', 'Precision', 'Recall', 'F1-score', 'Rand-Index'])
            writer.writerows(all_results)

        print(f'All results for {dataset} saved in results_{dataset}.csv')




