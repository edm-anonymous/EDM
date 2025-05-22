import torch 
import torch.nn as nn
import torch.nn.functional as F

from gcn import * 
from utils import * 

class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)

class GraphModel(nn.Module):
    def __init__(self, n_dim, nhidden, dropout, window_spk, window_dir, window_spk_f, window_dir_f, use_residue, epsilon=1, epsilon2=1,
                n_speakers=2, modals=['a','v','l'], use_speaker=True, use_modal=False, num_L=3, num_K=4, multimodal_node='both', graph_type='both',
                directed_edge='avl', single_edge=False):
        super(GraphModel, self).__init__()
        self.use_residue = use_residue
        self.dropout = dropout
        self.window_spk = window_spk
        self.window_dir = window_dir
        self.window_spk_f = window_spk_f
        self.window_dir_f = window_dir_f
        self.modals = modals
        self.modal_embeddings = nn.Embedding(3, n_dim)
        self.speaker_embeddings = nn.Embedding(n_speakers, n_dim)
        self.use_speaker = use_speaker
        self.use_modal = use_modal
        self.epsilon = epsilon
        self.epsilon2 = epsilon2  
        self.fc1 = nn.Linear(n_dim, nhidden)      
        self.fc2 = nn.Linear(n_dim, nhidden)   
        self.num_L =  num_L
        self.num_K =  num_K
        self.multimodal_node = multimodal_node
        self.graph_type = graph_type
        self.directed_edge = directed_edge
        self.single_edge = single_edge

        for ll in range(num_L):
            setattr(self,'directconv%d' %(ll+1), directConv(nhidden))
        for kk in range(num_K):
            setattr(self,'conv%d' %(kk+1), MultiConv(nhidden))

    def forward(self, a, v, l, u, dia_len, qmask):
        qmask_no_pad = torch.cat([qmask[:x,i,:] for i,x in enumerate(dia_len)],dim=0)
        spk_idx = torch.argmax(qmask_no_pad, dim=-1)
        spk_emb_vector = self.speaker_embeddings(spk_idx) 

        if self.use_modal:  
            emb_idx = torch.LongTensor([0, 1, 2]).cuda()
            emb_vector = self.modal_embeddings(emb_idx)
            if 'a' in self.modals:
                a += emb_vector[0].reshape(1, -1).expand(a.shape[0], a.shape[1])
            if 'v' in self.modals:
                v += emb_vector[1].reshape(1, -1).expand(v.shape[0], v.shape[1])
            if 'l' in self.modals:
                l += emb_vector[2].reshape(1, -1).expand(l.shape[0], l.shape[1])

    
        if 'l' in self.modals:
            tmp_l = l
        else:
            tmp_l = []
        if 'a' in self.modals:
            tmp_a = a
        else:
            tmp_a = []
        if 'v' in self.modals:
            tmp_v = v
        else:
            tmp_v = []

        tmp_u = u

        if self.use_speaker:
            if 'b' in self.use_speaker:
                if 'l' in self.modals:
                    tmp_l += spk_emb_vector
            if 'c' in self.use_speaker:
                if 'a' in self.modals:
                    tmp_a += spk_emb_vector
            if 'd' in self.use_speaker:
                if 'v' in self.modals:
                    tmp_v += spk_emb_vector
            if 'e' in self.use_speaker:
                tmp_u += spk_emb_vector
        if self.multimodal_node == 'both' or self.multimodal_node == 'ECG': 
            gnn_edge_index, cls_features = create_ECGNN_edges(tmp_a, tmp_v, tmp_l, tmp_u, dia_len, self.modals, self.window_dir, self.window_dir_f, self.directed_edge, self.single_edge)
        else:
            gnn_edge_index, cls_features = create_ECGNN_edges(tmp_a, tmp_v, tmp_l, None, dia_len, self.modals, self.window_dir, self.window_dir_f, self.directed_edge, self.single_edge)
        gnn_out = self.fc2(cls_features)
        initial_feat = gnn_out
        for kk in range(self.num_K):
            gnn_out = getattr(self,'conv%d' %(kk+1))(gnn_out,gnn_edge_index) + self.epsilon*initial_feat 

        if 'l' in self.modals:
                tmp_l2 = l
        else:
            tmp_l2 = []
        if 'a' in self.modals:
            tmp_a2 = a
        else:
            tmp_a2 = []
        if 'v' in self.modals:
            tmp_v2 = v
        else:
            tmp_v2 = []
        tmp_u2 = u

        if self.use_speaker:
            if 'f' in self.use_speaker:
                if 'l' in self.modals:
                    tmp_l2 += spk_emb_vector
            if 'g' in self.use_speaker:
                if 'a' in self.modals:
                    tmp_a2 += spk_emb_vector
            if 'h' in self.use_speaker:
                if 'v' in self.modals:
                    tmp_v2 += spk_emb_vector
            if 'i' in self.use_speaker:
                tmp_u2 += spk_emb_vector
        
        if self.multimodal_node == 'both' or self.multimodal_node == 'EIG': 
            speaker_edge_index, features = create_EIGNN_edges(tmp_a2, tmp_v2, tmp_l2, tmp_u2, dia_len, self.modals, qmask, self.window_spk, self.window_spk_f, self.directed_edge, self.single_edge)     
        else:
            speaker_edge_index, features = create_EIGNN_edges(tmp_a2, tmp_v2, tmp_l2, None, dia_len, self.modals, qmask, self.window_spk, self.window_spk_f, self.directed_edge, self.single_edge)     
        x1 = self.fc1(features)  
        initial_feat2 = x1
        out = x1
        for ll in range(self.num_L):
            out = getattr(self,'directconv%d' %(ll+1))(out, speaker_edge_index) + self.epsilon2*initial_feat2
        
        out1 = create_multimodal_feature(dia_len, out, gnn_out, len(self.modals), self.multimodal_node, self.graph_type) 
        
        return out1, spk_idx