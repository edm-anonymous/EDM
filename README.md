# Emotion Dynamics Modeling via Multimodal Fusion Graphs for Emotion Recognition in Conversation

## 🛠️ Requirements

- Python 3.8.5
- PyTorch 1.7.1
- CUDA 11.3
- torch-geometric 1.7.2
- fvcore, thop, torchinfo

## 📁 Datasets
The pre-extracted multimodal features (text, audio, visual) used in this project are adopted from the [M3Net](https://github.com/feiyuchen7/M3NET) project(Chen et al., CVPR 2023). Download multimodal features:
- [IEMOCAP](https://drive.google.com/drive/folders/1s5S1Ku679nlVZQPEfq-6LXgoN1K6Tzmz?usp=drive_link) → Place into the `IEMOCAP_features/` folder  
- [MELD](https://drive.google.com/drive/folders/1GfqY7WNVeCBWoFa_NSTalnaIgyyOVJuC?usp=drive_link) → Place into the `MELD_features/` folder


Download raw datasets(Optional):
- [IEMOCAP](https://sail.usc.edu/iemocap/)
- [MELD](https://github.com/SenticNet/MELD)

## 🏋️‍♀️ Training

### Train on IEMOCAP
```bash
python -u train.py --base-model 'GRU' --dropout 0.5 --lr 0.0001 --batch-size=32 --epochs=60 --multi_modal --mm_fusion_mthd='concat_DHT' --modals='avl' --Dataset='IEMOCAP' --norm BN --num_L=6 --num_K=3 --window_spk=10 --window_spk_f=-1 --window_dir=1 --window_dir_f=-1 --epsilon2=1 --epsilon=1 --use_speaker='bh' --multimodal_node='both' --graph_type='both' --directed_edge='avl' --single_edge=''
```

### Train on MELD
```bash
python -u train.py --base-model 'GRU' --dropout=0.4 --lr=0.0001 --batch-size 32 --epochs=6 --multi_modal --mm_fusion_mthd='concat_DHT' --modals='avl' --Dataset='MELD' --norm BN --num_L=1 --num_K=1 --window_spk=3 --window_spk_f=1 --window_dir=8 --window_dir_f=6 --epsilon2=0.1 --epsilon=1.1 --use_speaker='i' --multimodal_node='both' --graph_type='both' --directed_edge='avl' --single_edge=''
```

## 🚀 Quick Start
We also provide the best model checkpoints of our EDM for each dataset. Download checkpoints:
- [IEMOCAP](https://drive.google.com/file/d/1RGmLqOcXkLHCv8ibTHVYSHZa9aFoTH64/view?usp=drive_link)  
- [MELD](https://drive.google.com/file/d/1wy9mxnGHL1Mkt4napDzdoefe1MCQY6SL/view?usp=drive_link)
  
### Inference with Pretrained Checkpoint on IEMOCAP
```bash
python -u train.py --base-model 'GRU' --dropout 0.5 --lr 0.0001 --batch-size=32 --epochs=60 --multi_modal --mm_fusion_mthd='concat_DHT' --modals='avl' --Dataset='IEMOCAP' --norm BN --num_L=6 --num_K=3 --window_spk=10 --window_spk_f=-1 --window_dir=1 --window_dir_f=-1 --epsilon2=1 --epsilon=1 --use_speaker='bh' --multimodal_node='both' --graph_type='both' --directed_edge='avl' --single_edge='' --testing
```
> Checkpoint path: `./best_model_IEMOCAP.pth`

### Inference with Pretrained Checkpoint on MELD
```bash
python -u train.py --base-model 'GRU' --dropout=0.4 --lr=0.0001 --batch-size 32 --epochs=6 --multi_modal --mm_fusion_mthd='concat_DHT' --modals='avl' --Dataset='MELD' --norm BN --num_L=1 --num_K=1 --window_spk=3 --window_spk_f=1 --window_dir=8 --window_dir_f=6 --epsilon2=0.1 --epsilon=1.1 --use_speaker='i' --multimodal_node='both' --graph_type='both' --directed_edge='avl' --single_edge='' --testing
```
> Checkpoint path: `./best_model_MELD.pth`

## 📊 Evaluation & Export

- Accuracy, F1-score, classification report, and confusion matrix will be printed.
- It also saves intermediate features for analysis:
  in ./saved_features
  - `multimodal_emotions_before_<DATASET>.pkl`: Multimodal emotion representations before training
  - `multimodal_emotions_after_<DATASET>.pkl`: Multimodal emotion representations after training
  - `emotion_labels_<DATASET>.pkl`: Ground-truth emotion labels per utterance
  - `speaker_index_<DATASET>.pkl`: Speaker index per utterance
duin ./saved_outputs
  - `utterances_<DATASET>.pkl`: A CSV file with detailed utterance-level information `dialogue_id`, `utterance_idx`, `speaker`, `label`, `pred`

## 📉 Experiments
### Ablation Study
Each line shows the arguments changed to disable a specific model component or modality.
```bash
# Table 4 (i) w/o Early-Fusion Multimodal Node
--directed_edge=''
# Table 4 (ii) w/o EIGNN
--graph_type='ECG'
# Table 4 (iii) w/o ECGNN
--graph_type='EIG'
# Appendix Table 3 (i) Unimodal
Remove --multi_modal
--modals='l' or --modals='a' or --modals='v'
# Appendix Table 3 (ii) Bimodal
--modals='la' or --modals='lv' or --modals='av'
# Appendix Table 3 (iii) w/o Speaker Embedding
--use_speaker=''
# Appendix Table 4 (i) Intra-Utterance Directed Edge
--directed_edge='l' or --directed_edge='a' or --directed_edge='v' or --directed_edge='la' or --directed_edge='lv' or --directed_edge='av'
# Appendix Table 4 (ii) Intra-Utterance Undirected Multi-edge
--single_edge='intra' 
# Appendix Table 4 (iii) Inter-Utterance Undirected Multi-edge
--single_edge='inter'
```

### Effect of Hyperparameter
We also evaluate the model’s sensitivity to key hyperparameters:
```bash
# Figure 4: Number of GNN Layers
--num_L=<int>
--num_K=<int>
# Appendix: Effect of Window Size
# For EIGNN
--window_spk=<int>
--window_spk_f=<int>
# For ECGNN
--window_dir=<int>
--window_dir_f=<int>
# Appendix Figure 2: Effect of εI and εC
# For EIGNN
--epsilon2=<float>
# For ECGNN
--epsilon=<float>
```

### Additional Analyses after Train
```bash
# Figure 6: t-SNE visualization on IEMOCAP
python tsne_IEMOCAP.py
# Appendix Figure 3: t-SNE visualization on MELD
python tsne_MELD.py
# Appendix: Effect of modeling emotional inertia
python Inertia_IEMOCAP.py
python Inertia_MELD.py
# Appendix Table 6: FLOP, Memory, and Inference Time
--testing --overhead
``` 


## 🔧 Argument Highlights

| Argument              | Description                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| `--modals`            | Modalities to use for fusion: `'a'` (audio), `'v'` (visual), `'l'` (language). Use combinations like `'avl'`. |
| `--multi_modal`       | Enable multimodal input using text, audio, and visual features.             |
| `--num_L`             | Number of EIGNN layers.                           |
| `--num_K`             | Number of ECGNN layers.                       |
| `--window_spk`        | Past window size for EIGNN.                           |
| `--window_spk_f`      | Future window size for EIGNN.                         |
| `--window_dir`        | Past window size for ECGNN.                           |
| `--window_dir_f`      | Future window size for ECGNN.                         |
| `--epsilon`           | Scaling factor for ECGNN.                           |
| `--epsilon2`          | Scaling factor for EIGNN.                         |
| `--use_speaker`       | Type of speaker embedding in the graph.                         |
| `--multimodal_node`   | Multimodal node in the model: `'EIG'`, `'ECG'`, or `'both'`. |
| `--graph_type`        | Type of graph in the model: `'EIG'`, `'ECG'`, or `'both'`. |
| `--directed_edge`     | Applies directed intra-utterance edges to the multimodal node for selected modalities: `'a'`, `'v'`, `'l'`. |
| `--single_edge`       | Use only a single edge type: `'inter'` (inter-utterance) or `'intra'` (intra-utterance). |
| `--testing`           | Run in test mode using a pre-trained model checkpoint.                      |
| `--overhead`          | Run FLOP/memory/time analysis (only valid when `--testing` is used).        |
---
