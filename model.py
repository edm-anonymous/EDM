import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from model_graph import GraphModel

def print_grad(grad):
    print('the grad is', grad[2][0:5])
    return grad

class FocalLoss(nn.Module):
    def __init__(self, gamma = 2.5, alpha = 1, size_average = True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
    
    def forward(self, logits, labels):
        if labels.dim() > 2:
            labels = labels.contiguous().view(labels.size(0), labels.size(1), -1)
            labels = labels.transpose(1, 2)
            labels = labels.contiguous().view(-1, labels.size(2)).squeeze()
        if logits.dim() > 3:
            logits = logits.contiguous().view(logits.size(0), logits.size(1), logits.size(2), -1)
            logits = logits.transpose(2, 3)
            logits = logits.contiguous().view(-1, logits.size(1), logits.size(3)).squeeze()
        labels_length = logits.size(1)
        seq_length = logits.size(0)

        new_label = labels.unsqueeze(1)
        label_onehot = torch.zeros([seq_length, labels_length]).cuda().scatter_(1, new_label, 1)

        log_p = F.log_softmax(logits,-1)
        pt = label_onehot * log_p
        sub_pt = 1 - pt
        fl = -self.alpha * (sub_pt)**self.gamma * log_p
        if self.size_average:
            return fl.mean()
        else:
            return fl.sum()

class MaskedNLLLoss(nn.Module):

    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight,
                               reduction='sum')

    def forward(self, pred, target, mask):
        mask_ = mask.view(-1,1)
        if type(self.weight)==type(None):
            loss = self.loss(pred*mask_, target)/torch.sum(mask)
        else:
            loss = self.loss(pred*mask_, target)\
                            /torch.sum(self.weight[target]*mask_.squeeze())
        return loss


class MaskedMSELoss(nn.Module):

    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction='sum')

    def forward(self, pred, target, mask):
        loss = self.loss(pred*mask, target)/torch.sum(mask)
        return loss


class UnMaskedWeightedNLLLoss(nn.Module):

    def __init__(self, weight=None):
        super(UnMaskedWeightedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight,
                               reduction='sum')

    def forward(self, pred, target):
        if type(self.weight)==type(None):
            loss = self.loss(pred, target)
        else:
            loss = self.loss(pred, target)\
                            /torch.sum(self.weight[target])
        return loss

def pad(tensor, length, no_cuda):
    if isinstance(tensor, Variable):
        var = tensor
        if length > var.size(0):
            if not no_cuda:
                return torch.cat([var, torch.zeros(length - var.size(0), *var.size()[1:]).cuda()])
            else:
                return torch.cat([var, torch.zeros(length - var.size(0), *var.size()[1:])])
        else:
            return var
    else:
        if length > tensor.size(0):
            if not no_cuda:
                return torch.cat([tensor, torch.zeros(length - tensor.size(0), *tensor.size()[1:]).cuda()])
            else:
                return torch.cat([tensor, torch.zeros(length - tensor.size(0), *tensor.size()[1:])])
        else:
            return tensor


def simple_batch_graphify(features, lengths, no_cuda):
    node_features = []
    batch_size = features.size(1)
    for j in range(batch_size):
        node_features.append(features[:lengths[j], j, :])

    node_features = torch.cat(node_features, dim=0)  

    if not no_cuda:
        node_features = node_features.cuda()
    return node_features

class Model(nn.Module):

    def __init__(self, base_model, D_m, D_g, D_e, graph_hidden_size, n_speakers, window_spk, window_dir, window_spk_f, window_dir_f,
                 n_classes=7, dropout=0.5, no_cuda=False, use_residue=True, epsilon=1, epsilon2=1,D_m_t=1024,
                 D_m_v=512,D_m_a=100,modals='avl',att_type='gated',av_using_lstm=False, 
                 dataset='IEMOCAP', use_speaker=True, use_modal=False, norm='LN2', num_L = 3, num_K = 4, multimodal_node='both', graph_type='both', directed_edge='avl', single_edge=False
                 ):
        
        super(Model, self).__init__()

        self.base_model = base_model
        self.no_cuda = no_cuda
        self.dropout = dropout
        self.use_residue = use_residue
        self.modals = [x for x in modals]  
        self.use_speaker = use_speaker
        self.use_modal = use_modal
        self.att_type = att_type
        self.directed_edge = directed_edge
        self.single_edge = single_edge

        self.normBNa = nn.BatchNorm1d(D_m_t, affine=True)
        self.normBNb = nn.BatchNorm1d(D_m_t, affine=True)
        self.normBNc = nn.BatchNorm1d(D_m_t, affine=True)
        self.normBNd = nn.BatchNorm1d(D_m_t, affine=True)

        self.normLNa = nn.LayerNorm(D_m_t, elementwise_affine=True)
        self.normLNb = nn.LayerNorm(D_m_t, elementwise_affine=True)
        self.normLNc = nn.LayerNorm(D_m_t, elementwise_affine=True)
        self.normLNd = nn.LayerNorm(D_m_t, elementwise_affine=True)
        
        self.norm_strategy = norm
        if self.att_type == 'gated' or self.att_type == 'concat_subsequently' or self.att_type == 'concat_DHT':
            self.multi_modal = True
            self.av_using_lstm = av_using_lstm
        else:
            self.multi_modal = False
        self.dataset = dataset

        if self.base_model == 'LSTM':
            if not self.multi_modal:
                if len(self.modals) == 3:
                    hidden_ = 250
                elif ''.join(self.modals) == 'al':
                    hidden_ = 150
                elif ''.join(self.modals) == 'vl':
                    hidden_ = 150
                else:
                    hidden_ = 100
                self.linear_ = nn.Linear(D_m, hidden_)
                self.lstm = nn.LSTM(input_size=hidden_, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
            else:
                if 'a' in self.modals:
                    hidden_a = D_g
                    self.linear_a = nn.Linear(D_m_a, hidden_a)
                    if self.av_using_lstm:
                        self.lstm_a = nn.LSTM(input_size=hidden_a, hidden_size=D_g//2, num_layers=2, bidirectional=True, dropout=dropout)
                if 'v' in self.modals:
                    hidden_v = D_g
                    self.linear_v = nn.Linear(D_m_v, hidden_v)
                    if self.av_using_lstm:
                        self.lstm_v = nn.LSTM(input_size=hidden_v, hidden_size=D_g//2, num_layers=2, bidirectional=True, dropout=dropout)
                if 'l' in self.modals:
                    hidden_l = D_g
                    self.linear_l = nn.Linear(D_m_t, hidden_l)
                    self.lstm_l = nn.LSTM(input_size=hidden_l, hidden_size=D_g//2, num_layers=2, bidirectional=True, dropout=dropout)

        elif self.base_model == 'GRU':
            if 'a' in self.modals:
                hidden_a = D_g
                self.linear_a = nn.Linear(D_m_a, hidden_a)
                if self.av_using_lstm:
                    self.gru_a = nn.GRU(input_size=hidden_a, hidden_size=D_g//2, num_layers=2, bidirectional=True, dropout=dropout)
            if 'v' in self.modals:
                hidden_v = D_g
                self.linear_v = nn.Linear(D_m_v, hidden_v)
                if self.av_using_lstm:
                    self.gru_v = nn.GRU(input_size=hidden_v, hidden_size=D_g//2, num_layers=2, bidirectional=True, dropout=dropout)
            if 'l' in self.modals:
                hidden_l = D_g
                self.linear_l = nn.Linear(D_m_t, hidden_l)
                self.gru_l = nn.GRU(input_size=hidden_l, hidden_size=D_g//2, num_layers=2, bidirectional=True, dropout=dropout)
        
        elif self.base_model == 'Transformer':
            if 'a' in self.modals:
                hidden_a = D_g
                self.linear_a = nn.Linear(D_m_a, hidden_a)
                self.trans_a = nn.TransformerEncoderLayer(d_model=hidden_a, nhead=4)
            if 'v' in self.modals:
                hidden_v = D_g
                self.linear_v = nn.Linear(D_m_v, hidden_v)
                self.trans_v = nn.TransformerEncoderLayer(d_model=hidden_v, nhead=4)
            if 'l' in self.modals:
                hidden_l = D_g
                self.linear_l = nn.Linear(D_m_t, hidden_l)
                self.trans_l = nn.TransformerEncoderLayer(d_model=hidden_l, nhead=4)

        elif self.base_model == 'None':
            self.base_linear = nn.Linear(D_m, 2*D_e)

        else:
            print ('Base model must be one of DialogRNN/LSTM/GRU')
            raise NotImplementedError 

        self.linear_u = nn.Linear(D_g*len(self.modals), D_g)
        self.graph_model = GraphModel(n_dim=D_g, nhidden=graph_hidden_size, dropout=self.dropout, window_spk=window_spk, window_dir=window_dir, window_spk_f=window_spk_f, window_dir_f=window_dir_f,
                                    use_residue=self.use_residue, epsilon=epsilon, epsilon2=epsilon2, n_speakers=n_speakers, modals=self.modals, use_speaker=self.use_speaker, use_modal=self.use_modal, 
                                    num_L=num_L, num_K=num_K, multimodal_node=multimodal_node, graph_type=graph_type, directed_edge=directed_edge, single_edge=single_edge)
        if self.multi_modal:
            self.dropout_ = nn.Dropout(self.dropout)
            self.hidfc = nn.Linear(graph_hidden_size, n_classes)
            
            if self.use_residue:
                self.smax_fc = nn.Linear((D_g*2+graph_hidden_size*2), n_classes)
            else:
                if multimodal_node == 'both':
                    self.smax_fc = nn.Linear((graph_hidden_size*2), n_classes)
                else:
                    self.smax_fc = nn.Linear(((graph_hidden_size*2)*2), n_classes) 
                if graph_type != 'both':
                    self.smax_fc = nn.Linear((graph_hidden_size), n_classes)

    def _reverse_seq(self, X, mask):
        X_ = X.transpose(0,1)
        mask_sum = torch.sum(mask, 1).int()

        xfs = []
        for x, c in zip(X_, mask_sum):
            xf = torch.flip(x[:c], [0])
            xfs.append(xf)

        return pad_sequence(xfs)


    def forward(self, U, qmask, seq_lengths, U_a=None, U_v=None):
        if 'l' in self.modals:
            [r1,r2,r3,r4]=U 
            seq_len, _, feature_dim = r1.size()
            if self.norm_strategy == 'LN':
                r1 = self.normLNa(r1.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
                r2 = self.normLNb(r2.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
                r3 = self.normLNc(r3.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
                r4 = self.normLNd(r4.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
            elif self.norm_strategy == 'BN':
                r1 = self.normBNa(r1.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
                r2 = self.normBNb(r2.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
                r3 = self.normBNc(r3.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
                r4 = self.normBNd(r4.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
            elif self.norm_strategy == 'LN2':
                norm2 = nn.LayerNorm((seq_len, feature_dim), elementwise_affine=False)
                r1 = norm2(r1.transpose(0, 1)).transpose(0, 1)
                r2 = norm2(r2.transpose(0, 1)).transpose(0, 1)
                r3 = norm2(r3.transpose(0, 1)).transpose(0, 1)
                r4 = norm2(r4.transpose(0, 1)).transpose(0, 1)
            else:
                pass
            U = (r1 + r2 + r3 + r4)/4
        if len(self.modals) == 3:
            U_text_temp = r2
            U_audio_temp = U_a
            U_video_temp = U_v
            min_len = min(U_text_temp.size(0), U_audio_temp.size(0), U_video_temp.size(0))
            U_text_temp = U_text_temp[:min_len]
            U_audio_temp = U_audio_temp[:min_len]
            U_video_temp = U_video_temp[:min_len]
            emotions_u_temp = torch.cat([U_text_temp, U_audio_temp, U_video_temp], dim=-1)
            emotions_u_temp = simple_batch_graphify(emotions_u_temp, seq_lengths, self.no_cuda)

        if self.base_model == 'LSTM':
            if not self.multi_modal:
                U = self.linear_(U)
                emotions, _ = self.lstm(U)
            else:
                if 'a' in self.modals:
                    U_a = self.linear_a(U_a)
                    if self.av_using_lstm:
                        emotions_a, _ = self.lstm_a(U_a)
                    else:
                        emotions_a = U_a
                if 'v' in self.modals:
                    U_v = self.linear_v(U_v)
                    if self.av_using_lstm:
                        emotions_v, _ = self.lstm_v(U_v)
                    else:
                        emotions_v = U_v
                if 'l' in self.modals:
                    U = self.linear_l(U)
                    emotions_l, _ = self.lstm_l(U)
        elif self.base_model == 'GRU':
            if 'a' in self.modals:
                U_a = self.linear_a(U_a)
                if self.av_using_lstm:
                    emotions_a, _ = self.gru_a(U_a)
                else:
                    emotions_a = U_a
            if 'v' in self.modals:
                U_v = self.linear_v(U_v)
                if self.av_using_lstm:
                    emotions_v, _ = self.gru_v(U_v)
                else:
                    emotions_v = U_v
            if 'l' in self.modals:
                if self.dataset=='MELD':
                    pass
                else:
                    U = self.linear_l(U)
                emotions_l, _ = self.gru_l(U)
        
        if len(self.modals) ==3:
            min_len = min(emotions_l.size(0), emotions_a.size(0), emotions_v.size(0))
            emotions_l = emotions_l[:min_len]
            emotions_a = emotions_a[:min_len]
            emotions_v = emotions_v[:min_len]
            emotions_u = self.linear_u(torch.cat([emotions_l, emotions_a, emotions_v], dim=-1))
            

        elif len(self.modals) ==2:
            if 'a' not in self.modals:
                emotions_u = self.linear_u(torch.cat([emotions_l, emotions_v], dim=-1))
            elif 'l' not in self.modals:
                emotions_u = self.linear_u(torch.cat([emotions_a, emotions_v], dim=-1))
            elif 'v' not in self.modals:
                emotions_u = self.linear_u(torch.cat([emotions_l, emotions_a], dim=-1))
        else: 
            if 'l' in self. modals:
                emotions_u = self.linear_u(torch.cat([emotions_l], dim=-1))
            elif 'a' in self. modals:
                emotions_u = self.linear_u(torch.cat([emotions_a], dim=-1))
            elif 'v' in self. modals:
                emotions_u = self.linear_u(torch.cat([emotions_v], dim=-1))

        if not self.multi_modal:
            _ = simple_batch_graphify(emotions, seq_lengths, self.no_cuda)
        else:
            if 'a' in self.modals:
                features_a = simple_batch_graphify(emotions_a, seq_lengths, self.no_cuda)
            else:
                features_a = []
            if 'v' in self.modals:
                features_v = simple_batch_graphify(emotions_v, seq_lengths, self.no_cuda)
            else:
                features_v = []
            if 'l' in self.modals:
                features_l = simple_batch_graphify(emotions_l, seq_lengths, self.no_cuda)
            else:
                features_l = []
        features_u = simple_batch_graphify(emotions_u, seq_lengths, self.no_cuda)
        emotions_feat, spk_idx = self.graph_model(features_a, features_v, features_l, features_u, seq_lengths, qmask) 
        log_prob = F.log_softmax(self.smax_fc(emotions_feat), 1)
        return log_prob, emotions_feat, spk_idx, emotions_u_temp