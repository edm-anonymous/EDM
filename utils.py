import torch
import numpy as np
from itertools import permutations
import scipy.sparse as sp
from numpy import inf

def create_multimodal_feature(dia_len, out, gnn_out, num_modal, multimodal_node, graph_type):
    l, a, v, u = [], [], [], []
    if graph_type != 'both': 
        if graph_type == 'EIG':
            features = out
        elif graph_type == 'ECG':
            features = gnn_out
        for i in dia_len:
            if num_modal == 3:
                uu = features[3*i:4*i]
                features = features[4*i:]
            elif num_modal == 2:
                uu = features[2*i:3*i]
                features = features[3*i:]
            else:
                uu = features[1*i:2*i]
                features = features[2*i:]
            u.append(uu)
        return torch.cat(u,dim=0)

    if multimodal_node == 'both':
        features = torch.cat([out,gnn_out], dim=1)
        for i in dia_len:
            if num_modal == 3:
                uu = features[3*i:4*i]
                features = features[4*i:]
            elif num_modal == 2:
                uu = features[2*i:3*i]
                features = features[3*i:]
            else:
                uu = features[1*i:2*i]
                features = features[2*i:]
            u.append(uu)
        return torch.cat(u,dim=0)
    else:
        if multimodal_node == 'ECG':
            features = out
            features_2 = gnn_out
        else:
            features = gnn_out
            features_2 = out
        for i in dia_len:
            if num_modal == 3:
                m1 = features[0:1*i]
                m2 = features[1*i:2*i]
                m3 = features[2*i:3*i]
                uu = features_2[3*i:4*i]
                features = features[3*i:]
            elif num_modal == 2:
                m1 = features[0:1*i]
                m2 = features[1*i:2*i]
                uu = features_2[2*i:3*i]
                features = features[2*i:]
            else:
                m1 = features[0:1*i]
                uu = features_2[1*i:2*i]
                features = features[1*i:]
            l.append(m1)
            a.append(m2)
            v.append(m3)
            u.append(uu)
        tmpl = torch.cat(l,dim=0)
        tmpa = torch.cat(a,dim=0)
        tmpv = torch.cat(v,dim=0)
        tmpu = torch.cat(u,dim=0)
        multimodal_feature = torch.cat([tmpl, tmpa, tmpv, tmpu], dim=-1)

        return multimodal_feature


def create_EIGNN_edges(a, v, l, u, dia_len, modals, qmask, window_spk, window_spk_f, directed_edge, single_edge):
    num_modality = len(modals)
    node_count = 0
    index = []
    tmp = []
    spk = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for dia_idx, i in enumerate(dia_len):
        spk_mask = qmask[:i, dia_idx, :] 
        spk_mask = spk_mask.cpu()
        spk_idx = torch.argmax(spk_mask, dim=-1) 
        spk_idx = spk_idx.tolist()
        spk_1, spk_2, spk_3, spk_u = {}, {}, {}, {}
        if u is None:
            nodes = list(range(i*(num_modality)))
        else:
            nodes = list(range(i*(num_modality+1)))
        nodes = [j + node_count for j in nodes]
        nodes_1, nodes_2, nodes_3 = [], [], []
        if num_modality ==3:
            nodes_1 = nodes[0:i*num_modality//num_modality]
            nodes_2 = nodes[i*num_modality//num_modality:i*num_modality*2//num_modality]
            if u is None:
                nodes_3 = nodes[i*num_modality*2//num_modality:]
            else:
                nodes_3 = nodes[i*num_modality*2//num_modality:i*num_modality]
        elif num_modality ==2:
            nodes_1 = nodes[0:i*num_modality//num_modality]
            if u is None:
                nodes_2 = nodes[i*num_modality//num_modality:]
            else:
                nodes_2 = nodes[i*num_modality//num_modality:i*num_modality*2//num_modality]
        elif num_modality ==1:
            if u is None:
                nodes_1 = nodes[0:]
            else:
                nodes_1 = nodes[0:i*num_modality//num_modality]
        if u is not None:
            nodes_u = nodes[i*num_modality:]
           
        for j in range(i):
            if spk_idx[j] in spk_1:
                spk_1[spk_idx[j]].append((nodes_1[j]))
                if num_modality >= 2:
                    spk_2[spk_idx[j]].append((nodes_2[j]))
                if num_modality ==3:
                    spk_3[spk_idx[j]].append((nodes_3[j]))       
                if u is not None:
                    spk_u[spk_idx[j]].append((nodes_u[j]))             
            else:
                spk_1[spk_idx[j]] = [(nodes_1[j])]
                if num_modality >=2:
                    spk_2[spk_idx[j]] = [(nodes_2[j])]
                if num_modality ==3:
                    spk_3[spk_idx[j]] = [(nodes_3[j])]                    
                if u is not None:
                    spk_u[spk_idx[j]] = [(nodes_u[j])]     
        # edge type 01: intra modal edge
        for idx in spk_1:
            for j in range(1, len(spk_1[idx])):
                if window_spk == -1:
                    for k in range(j-1, -1, -1):
                        index.append((spk_1[idx][j], spk_1[idx][k])) 
                        if num_modality >=2:
                            index.append((spk_2[idx][j], spk_2[idx][k])) 
                        if num_modality ==3:
                            index.append((spk_3[idx][j], spk_3[idx][k])) 
                        if u is not None:
                            index.append((spk_u[idx][j], spk_u[idx][k])) 

                else:
                    for k in range(j-1, max(j-window_spk-1, -1), -1):
                        index.append((spk_1[idx][j], spk_1[idx][k])) 
                        if num_modality >=2:
                            index.append((spk_2[idx][j], spk_2[idx][k])) 
                        if num_modality ==3:
                            index.append((spk_3[idx][j], spk_3[idx][k])) 
                        if u is not None:
                            index.append((spk_u[idx][j], spk_u[idx][k])) 

                if window_spk_f == -1:
                    for k in range(j+1, len(spk_1[idx])):
                        index.append((spk_1[idx][j], spk_1[idx][k])) 
                        if num_modality >=2:
                            index.append((spk_2[idx][j], spk_2[idx][k])) 
                        if num_modality ==3:
                            index.append((spk_3[idx][j], spk_3[idx][k])) 
                        if u is not None:
                            index.append((spk_u[idx][j], spk_u[idx][k])) 
                       
                else:
                    for k in range(j+1, min(j+window_spk_f+1, len(spk_1[idx]))):
                        index.append((spk_1[idx][j], spk_1[idx][k])) 
                        if num_modality >=2:
                            index.append((spk_2[idx][j], spk_2[idx][k])) 
                        if num_modality ==3:
                            index.append((spk_3[idx][j], spk_3[idx][k])) 
                        if u is not None:
                            index.append((spk_u[idx][j], spk_u[idx][k])) 

        # edge type 02: inter modal edge       
        Gnodes=[]
        for _ in range(i):
            if len(modals) == 3:
                Gnodes.append([nodes_1[_]] + [nodes_2[_]] + [nodes_3[_]])
            elif len(modals) == 2:
                Gnodes.append([nodes_1[_]] + [nodes_2[_]])
            else:
                Gnodes.append([nodes_1[_]])
                
        for ii, _ in enumerate(Gnodes):
            perm = list(permutations(_, 2))
            for p in perm:
                tmp.append(p)
                if single_edge != 'intra':
                    tmp.append(p)

        # Edge type 03: new inter modal edge; {t, a, v} --> u
        if u is not None:
            if len(directed_edge) > 0:
                for idx in range(i):
                    if len(directed_edge) == 3:
                        spk.append((nodes_1[idx], nodes_u[idx]))
                        spk.append((nodes_2[idx], nodes_u[idx]))
                        spk.append((nodes_3[idx], nodes_u[idx]))
                    elif len(directed_edge) >= 1:
                        for m in directed_edge:
                            if m in modals:
                                m_idx = modals.index(m)
                                if m_idx == 0:
                                    spk.append((nodes_1[idx], nodes_u[idx]))
                                if m_idx == 1:
                                    spk.append((nodes_2[idx], nodes_u[idx]))
                                if m_idx == 2:
                                    spk.append((nodes_3[idx], nodes_u[idx]))

        if node_count == 0:
            if 'l' in modals:
                ll = l[0:0+i]
            if 'a' in modals:
                aa = a[0:0+i]
            if 'v' in modals:
                vv = v[0:0+i]
            if u is not None:
                uu = u[0:0+i]
            if len(modals) == 3:
                if u is None:
                    features = torch.cat([ll,aa,vv],dim=0)
                else:
                    features = torch.cat([ll,aa,vv, uu],dim=0)
            elif len(modals) == 2:
                if u is None:
                    if 'l' in modals and 'v' in modals:
                        features = torch.cat([ll,vv],dim=0)
                    if 'l' in modals and 'a' in modals:
                        features = torch.cat([aa,ll],dim=0)
                    if 'a' in modals and 'v' in modals:
                        features = torch.cat([aa,vv],dim=0)
                else:
                    if 'l' in modals and 'v' in modals:
                        features = torch.cat([ll,vv,uu],dim=0)
                    if 'l' in modals and 'a' in modals:
                        features = torch.cat([aa,ll, uu],dim=0)
                    if 'a' in modals and 'v' in modals:
                        features = torch.cat([aa,vv, uu],dim=0)
            else:
                if u is None:
                    if 'l' in modals:
                        features = torch.cat([ll], dim=0)
                    if 'a' in modals:
                        features = torch.cat([aa], dim=0)
                    if 'v' in modals:
                        features = torch.cat([vv], dim=0)
                else:
                    if 'l' in modals:
                        features = torch.cat([ll,uu], dim=0)
                    if 'a' in modals:
                        features = torch.cat([aa,uu], dim=0)
                    if 'v' in modals:
                        features = torch.cat([vv,uu], dim=0)
            temp = 0+i
        else:
            if 'l' in modals:
                ll = l[temp:temp+i]
            if 'a' in modals:
                aa = a[temp:temp+i]
            if 'v' in modals:
                vv = v[temp:temp+i]
            if u is not None:
                uu = u[temp:temp+i]

            if len(modals) == 3:
                if u is None:
                    features_temp = torch.cat([ll,aa,vv],dim=0)
                else:
                    features_temp = torch.cat([ll,aa,vv, uu],dim=0)
            elif len(modals) == 2:
                if u is None:
                    if 'l' in modals and 'v' in modals:                    
                        features_temp = torch.cat([ll,vv],dim=0)
                    if 'l' in modals and 'a' in modals:
                        features_temp = torch.cat([aa,ll],dim=0)
                    if 'a' in modals and 'v' in modals:
                        features_temp = torch.cat([aa,vv],dim=0)
                else:
                    if 'l' in modals and 'v' in modals:
                        features_temp = torch.cat([ll,vv,uu],dim=0)
                    if 'l' in modals and 'a' in modals:
                        features_temp = torch.cat([aa,ll, uu],dim=0)
                    if 'a' in modals and 'v' in modals:
                        features_temp = torch.cat([aa,vv, uu],dim=0)
            else:
                if u is None:
                    if 'l' in modals:
                        features_temp = torch.cat([ll],dim=0)
                    if 'a' in modals:
                        features_temp = torch.cat([aa],dim=0)
                    if 'v' in modals:
                        features_temp = torch.cat([vv],dim=0)
                else:
                    if 'l' in modals:
                        features_temp = torch.cat([ll,uu],dim=0)
                    if 'a' in modals:
                        features_temp = torch.cat([aa,uu],dim=0)
                    if 'v' in modals:
                        features_temp = torch.cat([vv,uu],dim=0)
            features =  torch.cat([features,features_temp],dim=0)
            temp = temp+i
        if u is None:
            node_count = node_count + i*(num_modality)
        else:
            node_count = node_count + i*(num_modality+1)

    edge_index = torch.cat([torch.LongTensor(index).T,torch.LongTensor(tmp).T, torch.LongTensor(spk).T],1).cuda()
    
    return edge_index, features


def create_ECGNN_edges(a, v, l, u, dia_len, modals, window_dir, window_dir_f, directed_edge, single_edge):
    num_modality = len(modals)
    node_count = 0
    index = []
    tmp = []
    dir = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i in dia_len:
        index_t = []
        if u is None:
           nodes = list(range(i*(num_modality))) 
        else:
            nodes = list(range(i*(num_modality+1)))
        nodes = [j + node_count for j in nodes]
        if num_modality ==3:
            nodes_1 = nodes[0:i*num_modality//num_modality]
            nodes_2 = nodes[i*num_modality//num_modality:i*num_modality*2//num_modality]
            if u is None:
                nodes_3 = nodes[i*num_modality*2//num_modality:]
            else:
                nodes_3 = nodes[i*num_modality*2//num_modality:i*num_modality]
        elif num_modality ==2:
            nodes_1 = nodes[0:i*num_modality//num_modality]
            if u is None: 
                nodes_2 = nodes[i*num_modality//num_modality:]
            else:
                nodes_2 = nodes[i*num_modality//num_modality:i*num_modality*2//num_modality]
        else:
            if u is None: 
                nodes_1 = nodes[0:] 
            else:
                nodes_1 = nodes[0:i*num_modality//num_modality]
        if u is not None:
            nodes_u = nodes[i*num_modality:]

        # Edge type 01: intra modal edge
        if num_modality ==3:
            index_t = list(permutations(nodes_1,2)) + list(permutations(nodes_2,2)) + list(permutations(nodes_3,2))
        elif num_modality ==2:
            index_t = list(permutations(nodes_1,2)) + list(permutations(nodes_2,2))
        else: 
            index_t = list(permutations(nodes_1,2))
        if u is not None:
            index_u = list(permutations(nodes_u, 2))
        for (x, y) in index_t:
            if window_dir == -1:
                if x > y:
                    index.append((x, y))
                    if single_edge != 'inter':
                        index.append((y, x))
            elif x - y >= 1 and x - y <= window_dir:
                index.append((x, y))    
                if single_edge != 'inter':
                    index.append((y, x)) 
            if window_dir_f == -1:
                if x < y:
                    index.append((x, y))
                    if single_edge != 'inter':
                        index.append((y, x))
            elif x - y <= -1 and x - y >= - window_dir_f:
                index.append((x, y))
                if single_edge != 'inter':
                    index.append((y, x))
        if u is not None:
            for (x,y) in index_u:
                if window_dir == -1:
                    if x > y:
                        index.append((x, y))
                        if single_edge != 'inter':
                            index.append((y, x))
                elif x - y >= 1 and x - y <= window_dir:
                    index.append((x, y))    
                    if single_edge != 'inter':
                        index.append((y, x))
                if window_dir_f == -1:
                    if x < y:
                        index.append((x, y))
                        if single_edge != 'inter':
                            index.append((y, x))
                elif x - y <= -1 and x - y >= - window_dir_f:
                    index.append((x, y))
                    if single_edge != 'inter':
                        index.append((y, x))

        # Edge type 02: inter modal edge       
        Gnodes=[]
        for _ in range(i):
            if len(modals) == 3:
                Gnodes.append([nodes_1[_]] + [nodes_2[_]] + [nodes_3[_]])
            elif len(modals) == 2:
                Gnodes.append([nodes_1[_]] + [nodes_2[_]])
            else:
                Gnodes.append([nodes_1[_]])
                
        for ii, _ in enumerate(Gnodes):
            perm = list(permutations(_, 2))
            for p in perm:
                tmp.append(p)
                if single_edge != 'intra':
                    tmp.append(p)

        # Edge type 03: new inter modal edge; {t, a, v} --> u
        if u is not None:
            if len(directed_edge) > 0:
                for idx in range(i):
                    if len(directed_edge) == 3:
                        dir.append((nodes_1[idx], nodes_u[idx]))
                        dir.append((nodes_2[idx], nodes_u[idx]))
                        dir.append((nodes_3[idx], nodes_u[idx]))
                    elif len(directed_edge) >= 1:
                        for m in directed_edge:
                            if m in modals:
                                m_idx = modals.index(m)
                                if m_idx == 0:
                                    dir.append((nodes_1[idx], nodes_u[idx]))
                                if m_idx == 1:
                                    dir.append((nodes_2[idx], nodes_u[idx]))
                                if m_idx == 2:
                                    dir.append((nodes_3[idx], nodes_u[idx]))

                    
        if node_count == 0:
            if 'l' in modals:
                ll = l[0:0+i]
            if 'a' in modals:
                aa = a[0:0+i]
            if 'v' in modals:
                vv = v[0:0+i]
            if u is None:
                if len(modals) == 3:
                    features = torch.cat([ll,aa,vv],dim=0)
                elif len(modals) == 2:
                    if 'l' in modals and 'a' in modals:
                        features = torch.cat([ll,aa],dim=0)
                    elif 'l' in modals and 'v' in modals:
                        features = torch.cat([ll,vv],dim=0)
                    elif 'a' in modals and 'v' in modals:
                        features = torch.cat([aa,vv],dim=0)
                else:
                    if 'l' in modals:
                        features = torch.cat([ll],dim=0)
                    elif 'a' in modals:
                        features = torch.cat([aa],dim=0)
                    elif 'v' in modals:
                        features = torch.cat([vv],dim=0) 
            else:
                uu = u[0:0+i]
                if len(modals) == 3:
                    features = torch.cat([ll,aa,vv, uu],dim=0)
                elif len(modals) ==2:
                    if 'l' in modals and 'a' in modals:
                        features = torch.cat([ll,aa,uu],dim=0)
                    elif 'l' in modals and 'v' in modals:
                        features = torch.cat([ll,vv,uu],dim=0)
                    elif 'a' in modals and 'v' in modals:
                        features = torch.cat([aa,vv,uu],dim=0)
                else:
                    if 'l' in modals:
                        features = torch.cat([ll,uu],dim=0)
                    elif 'a' in modals:
                        features = torch.cat([aa,uu],dim=0)
                    elif 'v' in modals:
                        features = torch.cat([vv,uu],dim=0)
            temp = 0+i
        else:
            if 'l' in modals:
                ll = l[temp:temp+i]
            if 'a' in modals:
                aa = a[temp:temp+i]
            if 'v' in modals:
                vv = v[temp:temp+i]
            if u is None:
                if len(modals) == 3:
                    features_temp = torch.cat([ll,aa,vv],dim=0)
                elif len(modals) ==2:
                    if 'l' in modals and 'a' in modals:
                        features_temp = torch.cat([ll,aa],dim=0)
                    elif 'l' in modals and 'v' in modals:
                        features_temp = torch.cat([ll,vv],dim=0)
                    elif 'a' in modals and 'v' in modals:
                        features_temp = torch.cat([aa,vv],dim=0)
                else:
                    if 'l' in modals:
                        features_temp = torch.cat([ll],dim=0)
                    elif 'a' in modals:
                        features_temp = torch.cat([aa],dim=0)
                    elif 'v' in modals:
                        features_temp = torch.cat([vv],dim=0)
            else:
                uu = u[temp:temp+i]
                if len(modals) == 3:
                    features_temp = torch.cat([ll,aa,vv, uu],dim=0)
                elif len(modals) ==2:
                    if 'l' in modals and 'a' in modals:
                        features_temp = torch.cat([ll,aa,uu],dim=0)
                    elif 'l' in modals and 'v' in modals:
                        features_temp = torch.cat([ll,vv,uu],dim=0)
                    elif 'a' in modals and 'v' in modals:
                        features_temp = torch.cat([aa,vv,uu],dim=0)
                else:
                    if 'l' in modals:
                        features_temp = torch.cat([ll,uu],dim=0)
                    elif 'a' in modals:
                        features_temp = torch.cat([aa,uu],dim=0)
                    elif 'v' in modals:
                        features_temp = torch.cat([vv,uu],dim=0)

            features =  torch.cat([features,features_temp],dim=0)
            temp = temp+i
        if u is None:
            node_count = node_count + i*(num_modality)
        else:
            node_count = node_count + i*(num_modality+1)
    edge_index = torch.cat([torch.LongTensor(index).T,torch.LongTensor(tmp).T, torch.LongTensor(dir).T],1).cuda()

    return edge_index, features
