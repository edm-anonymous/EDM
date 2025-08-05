import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle, pandas as pd
import numpy

class IEMOCAPDataset(Dataset):

    def __init__(self, train=True, use_4way=False):
        self.use_4way = use_4way

        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid = pickle.load(open('./IEMOCAP_features/IEMOCAP_features.pkl', 'rb'), encoding='latin1')
        
        _, _, self.roberta1, self.roberta2, self.roberta3, self.roberta4,\
        _, _, _, _ = pickle.load(open("./IEMOCAP_features/iemocap_features_roberta.pkl", 'rb'), encoding='latin1')

        raw_keys = self.trainVid if train else self.testVid

        if use_4way:
            self.keys = []
            for k in raw_keys:
                labels = self.videoLabels[k]
                if all(l in [1, 2, 3, 5] for l in labels):  # neu, ang, sad, hap
                    self.keys.append(k)
        else:
            self.keys = list(raw_keys)

        self.len = len(self.keys)

    def remap_label(self, l):
        mapping = {
            1: 0,  
            2: 1,  
            3: 2,  
            5: 3   
        }
        return mapping[l]

    def __getitem__(self, index):
        vid = self.keys[index]
        labels = self.videoLabels[vid]
        if self.use_4way:
            labels = [self.remap_label(l) for l in labels]

        return torch.FloatTensor(numpy.array(self.roberta1[vid])),\
               torch.FloatTensor(numpy.array(self.roberta2[vid])),\
               torch.FloatTensor(numpy.array(self.roberta3[vid])),\
               torch.FloatTensor(numpy.array(self.roberta4[vid])),\
               torch.FloatTensor(numpy.array(self.videoVisual[vid])),\
               torch.FloatTensor(numpy.array(self.videoAudio[vid])),\
               torch.FloatTensor(numpy.array([[1,0] if x=='M' else [0,1] for x in self.videoSpeakers[vid]])),\
               torch.FloatTensor(numpy.array([1]*len(labels))),\
               torch.LongTensor(numpy.array(labels)),\
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<7 else pad_sequence(dat[i], True) if i<9 else dat[i].tolist() for i in dat]

class MELDDataset(Dataset):

    def __init__(self, path, train=True):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid, _ = pickle.load(open(path, 'rb'), encoding='latin1')

        _, _, _, self.roberta1, self.roberta2, self.roberta3, self.roberta4, \
        _, self.trainIds, self.testIds, self.validIds \
        = pickle.load(open("./MELD_features/meld_features_roberta.pkl", 'rb'), encoding='latin1') 


        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(numpy.array(self.roberta1[vid])),\
               torch.FloatTensor(numpy.array(self.roberta2[vid])),\
               torch.FloatTensor(numpy.array(self.roberta3[vid])),\
               torch.FloatTensor(numpy.array(self.roberta4[vid])),\
               torch.FloatTensor(numpy.array(self.videoVisual[vid])),\
               torch.FloatTensor(numpy.array(self.videoAudio[vid])),\
               torch.FloatTensor(numpy.array(self.videoSpeakers[vid])),\
               torch.FloatTensor(numpy.array([1]*len(self.videoLabels[vid]))),\
               torch.LongTensor(numpy.array(self.videoLabels[vid])),\
               int(vid)  

    def __len__(self):
        return self.len

    def return_labels(self):
        return_label = []
        for key in self.keys:
            return_label+=self.videoLabels[key]
        return return_label

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<7 else pad_sequence(dat[i], True) if i<9 else dat[i].tolist() for i in dat]

class CMUMOSEIDataset7(Dataset):

    def __init__(self, path='CMU-MOSEI_features/cmumosei_multi_regression_features.pkl', train=True):

        (
            self.videoIDs,
            self.videoSpeakers,
            self.videoLabels,
            self.videoText,
            self.videoAudio,
            self.videoVisual,
            self.videoSentence,
            self.trainVid,
            self.testVid,
        ) = pickle.load(open(path, "rb"), encoding="latin1")

        self.keys = self.trainVid if train else self.testVid

        self.len = len(self.keys)

        labels_emotion = {}
        for item in self.videoLabels:
            array = []
            for a in self.videoLabels[item]:
                if a < -2:
                    array.append(0)
                elif -2 <= a and a < -1:
                    array.append(1)
                elif -1 <= a and a < 0:
                    array.append(2)
                elif 0 <= a and a <= 0:
                    array.append(3)
                elif 0 < a and a <= 1:
                    array.append(4)
                elif 1 < a and a <= 2:
                    array.append(5)
                elif a > 2:
                    array.append(6)
            labels_emotion[item] = array
        self.labels_emotion = labels_emotion

        labels_sentiment = {}
        for item in self.videoLabels:
            array = []
            for a in self.videoLabels[item]:
                if a < 0:
                    array.append(0)
                elif 0 <= a and a <= 0:
                    array.append(1)
                elif a > 0:
                    array.append(2)
            labels_sentiment[item] = array
        self.labels_sentiment = labels_sentiment

    def __getitem__(self, index):
        vid = self.keys[index]
        return (
            torch.FloatTensor(numpy.array(self.videoText[vid])),
            torch.FloatTensor(numpy.array(self.videoText[vid])),
            torch.FloatTensor(numpy.array(self.videoText[vid])),
            torch.FloatTensor(numpy.array(self.videoText[vid])),
            torch.FloatTensor(numpy.array(self.videoVisual[vid])),
            torch.FloatTensor(numpy.array(self.videoAudio[vid])),
            torch.FloatTensor(
                [
                    [1, 0] if x == "M" else [0, 1]
                    for x in numpy.array(self.videoSpeakers[vid])
                ]
            ),
            torch.FloatTensor([1] * len(numpy.array(self.labels_emotion[vid]))),
            torch.LongTensor(numpy.array(self.labels_emotion[vid])),
            torch.LongTensor(numpy.array(self.labels_sentiment[vid])),
            vid,
        )

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)

        return [
            (
                pad_sequence(dat[i])
                if i < 7
                else pad_sequence(dat[i]) if i < 10 else dat[i].tolist()
            )
            for i in dat
        ]