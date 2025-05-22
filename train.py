import numpy as np, argparse, time, random
import numpy as np, argparse, time, random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dataloader import IEMOCAPDataset, MELDDataset
from model import Model, FocalLoss
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report
from fvcore.nn import FlopCountAnalysis, flop_count_table 
from thop import profile
from torchinfo import summary
import os
import pickle
import pandas as pd

seed = 67137 

def seed_everything(seed=seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
def _init_fn(worker_id):
    np.random.seed(int(seed)+worker_id)

def get_train_valid_sampler(trainset, valid=0.1, dataset='IEMOCAP'):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid*size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])

def get_MELD_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = MELDDataset('MELD_features/MELD_features_raw1.pkl')
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid, 'MELD')

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = MELDDataset('MELD_features/MELD_features_raw1.pkl', train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


def get_IEMOCAP_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = IEMOCAPDataset()
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory, worker_init_fn=_init_fn)

    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = IEMOCAPDataset(train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory, worker_init_fn=_init_fn)

    return train_loader, valid_loader, test_loader

def train_or_eval_graph_model(model, loss_function, dataloader, cuda, modals, optimizer=None, train=False): 
    losses, preds, labels = [], [], []
    multimodal_emotions_all = []
    multimodal_emotions_before_all = []
    spk_idx_all = []
    ei, et, en, el = torch.empty(0).type(torch.LongTensor), torch.empty(0).type(torch.LongTensor), torch.empty(0), []

    if cuda:
        ei, et, en = ei.cuda(), et.cuda(), en.cuda()

    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()
        
    seed_everything()

    for data in dataloader:
        if train:
            optimizer.zero_grad()

        textf1, textf2, textf3, textf4, visuf, acouf, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda else data[:-1]
        lengths = [(umask[j] == 1).nonzero(as_tuple=False).tolist()[-1][0] + 1 for j in range(len(umask))]
        # Supplementary - Ablation: (i, ii) Effect of Modality Combinations
        if args.multi_modal:
            if len(modals) == 3:
                log_prob, multimodal_emotions, spk_idx, multimodal_emotions_before = model([textf1,textf2,textf3,textf4], qmask, lengths, acouf, visuf)
            else:
                if 'a' not in modals:
                    log_prob, multimodal_emotions, spk_idx, multimodal_emotions_before = model([textf1,textf2,textf3,textf4], qmask, lengths, None, visuf)
                elif 'v' not in modals:
                    log_prob, multimodal_emotions, spk_idx, multimodal_emotions_before = model([textf1,textf2,textf3,textf4], qmask, lengths, acouf, None)
                else:
                    log_prob, multimodal_emotions, spk_idx, multimodal_emotions_before = model(None, qmask, lengths, acouf, visuf)
        else:
            if modals == 'a':
                log_prob, multimodal_emotions, spk_idx, multimodal_emotions_before = model(None, qmask, lengths, acouf, None)
            elif modals == 'v':
                log_prob, multimodal_emotions, spk_idx, multimodal_emotions_before = model(None, qmask, lengths, None, visuf)
            elif modals == 'l':
                log_prob, multimodal_emotions, spk_idx, multimodal_emotions_before = model([textf1,textf2,textf3,textf4], qmask, lengths)
        
        spk_idx_all.append(spk_idx.cpu().detach().numpy()) 
        multimodal_emotions_all.append(multimodal_emotions.cpu().detach().numpy())
        if multimodal_emotions_before != None: 
            multimodal_emotions_before_all.append(multimodal_emotions_before.cpu().detach().numpy())  
        label = torch.cat([label[j][:lengths[j]] for j in range(len(label))])
        loss = loss_function(log_prob, label)
        preds.append(torch.argmax(log_prob, 1).cpu().numpy())
        labels.append(label.cpu().numpy())
        losses.append(loss.item())
        if train:
            loss.backward()
            optimizer.step()
    if preds!=[]:
        preds  = np.concatenate(preds)
        labels = np.concatenate(labels)
    else:
        return float('nan'), float('nan'), [], [], float('nan'), _

    ei = ei.data.cpu().numpy()
    et = et.data.cpu().numpy()
    en = en.data.cpu().numpy()
    el = np.array(el)
    labels = np.array(labels)
    preds = np.array(preds)

    avg_loss = round(np.sum(losses)/len(losses), 4)
    avg_accuracy = round(accuracy_score(labels, preds)*100, 2)
    avg_fscore = round(f1_score(labels,preds, average='weighted')*100, 2)

    return avg_loss, avg_accuracy, labels, preds, avg_fscore, multimodal_emotions_all, spk_idx_all, multimodal_emotions_before_all

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='If set, disables GPU usage even if CUDA is available.')

    parser.add_argument('--base-model', default='LSTM',
                        help='Recurrent base model to use in feature encoding. Options: LSTM, GRU, TRANSFORMER.')

    parser.add_argument('--window_spk', type=int, default=1,
                        help='Size of the past speaker window used in EIGNN for temporal graph construction.')

    parser.add_argument('--window_dir', type=int, default=-1,
                        help='Size of the past directed window used in ECGNN for temporal edge construction.')

    parser.add_argument('--window_spk_f', type=int, default=1,
                        help='Size of the future speaker window used in EIGNN.')

    parser.add_argument('--window_dir_f', type=int, default=-1,
                        help='Size of the future directed window used in ECGNN.')

    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='Initial learning rate for the optimizer.')

    parser.add_argument('--l2', type=float, default=0.00003, metavar='L2',
                        help='L2 regularization weight (weight decay) to avoid overfitting.')

    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout',
                        help='Dropout rate applied in various model layers to prevent overfitting.')

    parser.add_argument('--batch-size', type=int, default=32, metavar='BS',
                        help='Number of samples per batch during training.')

    parser.add_argument('--epochs', type=int, default=60, metavar='E',
                        help='Total number of training epochs.')

    parser.add_argument('--class-weight', action='store_true', default=True,
                        help='If set, uses class weights in the loss function to address class imbalance.')

    parser.add_argument('--use_residue', action='store_true', default=False,
                        help='If set, adds residual connections in the graph layers.')

    parser.add_argument('--multi_modal', action='store_true', default=False,
                        help='Enables usage of multimodal inputs (text, audio, visual).')

    parser.add_argument('--mm_fusion_mthd', default='concat',
                        help="Fusion strategy for multimodal inputs. Options: 'concat', 'gated', 'concat_subsequently'.")

    parser.add_argument('--modals', default='avl',
                        help="Modalities to use for graph: 'a' for audio, 'v' for visual, 'l' for language. Use combinations like 'avl'.")

    parser.add_argument('--av_using_lstm', action='store_true', default=False,
                        help='If set, applies LSTM encoding to audio and visual features before fusion.')

    parser.add_argument('--Dataset', default='IEMOCAP',
                        help="Dataset to train and evaluate the model. Options: 'IEMOCAP' or 'MELD'.")

    parser.add_argument('--use_speaker', default='bcd',
                        help="Type of speaker embedding in the model")

    parser.add_argument('--use_modal', action='store_true', default=False,
                        help='If set, uses explicit modality embeddings.')

    parser.add_argument('--norm', default='LN2',
                        help="Normalization type used in the model. Options: 'BN', 'LN', 'LN2'.")

    parser.add_argument('--testing', action='store_true', default=False,
                        help='If set, runs in test mode using a pre-trained checkpoint.')

    parser.add_argument('--epsilon', type=float, default=1,
                        help='Scaling factor for ECGNN')

    parser.add_argument('--epsilon2', type=float, default=1,
                        help='Scaling factor for EIGNN')

    parser.add_argument('--num_L', type=int, default=3,
                        help='Number of EIGNN layers.')

    parser.add_argument('--num_K', type=int, default=4,
                        help='Number of ECGNN layers.')

    parser.add_argument('--overhead', action='store_true', default=False,
                        help='If set, reports FLOPs, memory usage, and inference time for performance analysis.')

    parser.add_argument('--multimodal_node', default='both',
                        help="Multimodal node used in the model. Options: 'EIG', 'ECG', 'both'.")
    
    parser.add_argument('--graph_type', default='both',
                        help="Type of graph used in the model. Options: 'EIG', 'ECG', 'both'.")

    parser.add_argument('--directed_edge', default='avl',
                        help="Applies directed intra-utterance edges to the multimodal node for selected modalities: 'a' for audio, 'v' for visual, 'l' for language. Use combinations like 'avl'.")

    parser.add_argument('--single_edge', default='',
                        help="Restrict to a single edge type. Options: 'inter' for inter-utterance, 'intra' for intra-utterance.")


    args = parser.parse_args()
    print(args)

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    cuda       = args.cuda
    n_epochs   = args.epochs
    batch_size = args.batch_size
    modals = args.modals

    feat2dim = {'IS10':1582,'3DCNN':512,'textCNN':100,'bert':768,'denseface':342,'MELD_text':600,'MELD_audio':300}
    if args.Dataset=='IEMOCAP':
        D_audio = feat2dim['IS10']  
    elif args.Dataset=='MELD': 
        D_audio = feat2dim['MELD_audio']
    
    if args.Dataset=='IEMOCAP' or args.Dataset=='MELD':
        D_visual = feat2dim['denseface']

    if args.Dataset=='IEMOCAP' or args.Dataset=='MELD':
        D_text = 1024

    if args.multi_modal:
        if args.mm_fusion_mthd=='concat':
            if modals == 'avl':
                D_m = D_audio+D_visual+D_text
            elif modals == 'av':
                D_m = D_audio+D_visual
            elif modals == 'al':
                D_m = D_audio+D_text
            elif modals == 'vl':
                D_m = D_visual+D_text
            else:
                raise NotImplementedError
        else:
            D_m = 1024
    else:
        if modals == 'a':
            D_m = D_audio
        elif modals == 'v':
            D_m = D_visual
        elif modals == 'l':
            D_m = D_text
        else:
            raise NotImplementedError
    D_g = 512 if args.Dataset=='IEMOCAP' else 1024
    D_e = 100
    graph_h = 512
    n_speakers = 9 if args.Dataset=='MELD' else 2
    
    if args.Dataset =='MELD':
        n_classes  = 7
    elif args.Dataset =='IEMOCAP':
        n_classes = 6
    else:
        n_classes = 1

    seed_everything()
    model = Model(args.base_model,
                D_m, D_g, D_e, graph_h,
                n_speakers=n_speakers,
                window_spk=args.window_spk,
                window_dir=args.window_dir,
                window_spk_f=args.window_spk_f,
                window_dir_f=args.window_dir_f,
                n_classes=n_classes,
                dropout=args.dropout,
                no_cuda=args.no_cuda,
                use_residue=args.use_residue,
                D_m_t = D_text,
                epsilon=args.epsilon,
                epsilon2=args.epsilon2,
                D_m_v = D_visual,
                D_m_a = D_audio,
                modals=args.modals,
                att_type=args.mm_fusion_mthd,
                av_using_lstm=args.av_using_lstm,
                dataset=args.Dataset,
                use_speaker=args.use_speaker,
                use_modal=args.use_modal,
                norm = args.norm,
                num_L = args.num_L,
                num_K = args.num_K,
                multimodal_node=args.multimodal_node,
                graph_type=args.graph_type,
                directed_edge=args.directed_edge,
                single_edge=args.single_edge)

    print ('Graph NN with', args.base_model, 'as base model.')
    name = 'Graph'

    if cuda:
        model.cuda()

    if args.Dataset == 'IEMOCAP':
        loss_weights = torch.FloatTensor([1/0.086747,
                                        1/0.144406,
                                        1/0.227883,
                                        1/0.160585,
                                        1/0.127711,
                                        1/0.252668])
        loss_function  = nn.NLLLoss(loss_weights.cuda() if cuda else loss_weights)

    elif args.Dataset == 'MELD':
        loss_function = FocalLoss()
        

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    lr = args.lr
    
    if args.Dataset == 'MELD':
        train_loader, valid_loader, test_loader = get_MELD_loaders(valid=0.0,
                                                                    batch_size=batch_size,
                                                                    num_workers=2)
    elif args.Dataset == 'IEMOCAP':
        train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(valid=0.0,
                                                                      batch_size=batch_size,
                                                                      num_workers=2)                                                       
    else:
        print("There is no such dataset")

    best_fscore, best_loss, best_label, best_pred, best_mask = None, None, None, None, None
    all_fscore, all_acc, all_loss = [], [], []
    
    if args.testing:
        # Supplementary - Time and Memory Overhead
        if args.overhead:
            batch_size = 1
            seq_len = 40 
            device = torch.device("cuda" if args.cuda else "cpu")

            dummy_text = tuple(torch.randn(seq_len, batch_size, 1024).to(device) for _ in range(4))
            dummy_qmask = torch.zeros(seq_len, batch_size, 9).to(device)  # max speakers
            dummy_lengths = [seq_len] * batch_size
            dummy_acouf = torch.randn(seq_len, batch_size, D_audio).to(device)
            dummy_visuf = torch.randn(seq_len, batch_size, D_visual).to(device)

            with torch.no_grad():
                inputs = (dummy_text, dummy_qmask, dummy_lengths, dummy_acouf, dummy_visuf)
                flops = FlopCountAnalysis(model, inputs)
                print(f"FLOPs: {flops.total() / 1e9:.2f} GFLOPs")

                allocated = torch.cuda.memory_allocated() / 1024**2
                reserved = torch.cuda.memory_reserved() / 1024**2
                print(f"Memory allocated: {allocated:.2f} MB")

                total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"Total Trainable Parameters: {total_params / 1e6:.2f}M")
                model.eval()
                for _ in range(10): 
                    _ = model(*inputs)

                torch.cuda.synchronize()
                start_time = time.time()
                _ = model(*inputs)
                torch.cuda.synchronize()
                end_time = time.time()

                elapsed_time_ms = (end_time - start_time) * 1000
                print(f"Inference Time (1 forward pass): {elapsed_time_ms:.3f} ms")


        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.Dataset == 'IEMOCAP':
            checkpoint_path='./best_model_IEMOCAP.pth'
        else:
           checkpoint_path='./best_model_MELD.pth' 
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        
        print('testing loaded model')
        test_loss, test_acc, test_label, test_pred, test_fscore, multimodal_emotions_after_train, spk_idx, multimodal_emotions_before_train = train_or_eval_graph_model(model, loss_function, test_loader, cuda, args.modals) 
        print('test_acc:',test_acc,'test_fscore:',test_fscore)

        print(classification_report(test_label, test_pred, sample_weight=best_mask,digits=4))
        print(confusion_matrix(test_label,test_pred,sample_weight=best_mask))
        n_epochs =0
        
        save_dir = "saved_features"  
        os.makedirs(save_dir, exist_ok=True)  

        if args.Dataset == 'MELD':
            with open(os.path.join(save_dir, "multimodal_emotions_before_MELD.pkl"), "wb") as f:
                pickle.dump(multimodal_emotions_before_train, f)
            with open(os.path.join(save_dir, "multimodal_emotions_after_MELD.pkl"), "wb") as f:
                pickle.dump(multimodal_emotions_after_train, f)
            with open(os.path.join(save_dir, "emotion_labels_MELD.pkl"), "wb") as f:
                pickle.dump(test_label, f)
            with open(os.path.join(save_dir, "speaker_index_MELD.pkl"), "wb") as f:
                pickle.dump(spk_idx, f)
        else:
            with open(os.path.join(save_dir, "multimodal_emotions_before_IEMOCAP.pkl"), "wb") as f:
                pickle.dump(multimodal_emotions_before_train, f)
            with open(os.path.join(save_dir, "multimodal_emotions_after_IEMOCAP.pkl"), "wb") as f:
                pickle.dump(multimodal_emotions_after_train, f)
            with open(os.path.join(save_dir, "emotion_labels_IEMOCAP.pkl"), "wb") as f:
                pickle.dump(test_label, f)
            with open(os.path.join(save_dir, "speaker_index_IEMOCAP.pkl"), "wb") as f:
                pickle.dump(spk_idx, f)
        
        qmask_all = []
        utt_lengths = []
        for data in test_loader:
            textf1, textf2, textf3, textf4, visuf, acouf, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda else data[:-1]
            lengths = [(umask[j] == 1).nonzero(as_tuple=False).tolist()[-1][0] + 1 for j in range(len(umask))]
            qmask_all.extend(torch.cat([qmask[:x,i,:] for i,x in enumerate(lengths)],dim=0))
            utt_lengths.extend(lengths)

        utterance_info = []
        flat_idx = 0
        for d_id, (qmask_d, length) in enumerate(zip(qmask_all, utt_lengths)):
            for u_idx in range(length):                
                speaker = int(torch.argmax(qmask_all[flat_idx]))
                label = int(test_label[flat_idx])
                pred = int(test_pred[flat_idx])
                utterance_info.append({
                    "dialogue_id": d_id,
                    "utterance_idx": u_idx,
                    "speaker": speaker,
                    "label": label,
                    "pred": pred
                })
                flat_idx += 1
        df = pd.DataFrame(utterance_info)

        os.makedirs("saved_outputs", exist_ok=True)
        if args.Dataset == 'MELD':
            df.to_csv("saved_outputs/utterances_MELD.csv", index=False)
        else:
            df.to_csv("saved_outputs/utterances_IEMOCAP.csv", index=False)
    

    for e in range(n_epochs):
        start_time = time.time()
        train_loss, train_acc, _, _, train_fscore, _, _, _ = train_or_eval_graph_model(model, loss_function, train_loader, cuda, args.modals, optimizer, True) 
        valid_loss, valid_acc, _, _, _, _ = train_or_eval_graph_model(model, loss_function, valid_loader, cuda, args.modals) 
        test_loss, test_acc, test_label, test_pred, test_fscore, _, _, _ = train_or_eval_graph_model(model, loss_function, test_loader, cuda, args.modals) 

        all_fscore.append(test_fscore)

        if best_loss == None or best_loss > test_loss:
            best_loss, best_label, best_pred = test_loss, test_label, test_pred

        if best_fscore == None or best_fscore < test_fscore:
            best_fscore = test_fscore
            best_label, best_pred = test_label, test_pred
            if args.Dataset == 'IEMOCAP' and test_fscore >= 74.2:
                torch.save(model.state_dict(), "./best_model_IEMOCAP.pth")
                print('test_fscore:', test_fscore)
            elif args.Dataset =='MELD' and test_fscore >= 67.4:
                torch.save(model.state_dict(), "./best_model_MELD.pth")
                print('test_fscore:', test_fscore)
       
        print('epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'.\
                format(e+1, train_loss, train_acc, train_fscore, test_loss, test_acc, test_fscore, round(time.time()-start_time, 2)))
        if (e+1)%10 == 0:
            print ('----------best F-Score:', max(all_fscore))
            print(classification_report(best_label, best_pred, sample_weight=best_mask,digits=4))
            print(confusion_matrix(best_label,best_pred,sample_weight=best_mask))
    
    if not args.testing:
        print('Test performance..')
        print ('F-Score:', max(all_fscore))
        print(classification_report(best_label, best_pred, sample_weight=best_mask,digits=4))
        print(confusion_matrix(best_label,best_pred,sample_weight=best_mask))