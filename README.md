# Emotion Dynamics Modeling via Multimodal Fusion Graphs for Emotion Recognition in Conversation

## 🛠️ Requirements

- Python 3.8.5
- PyTorch 1.7.1
- CUDA 11.3
- torch-geometric 1.7.2
- fvcore, thop, torchinfo

## 📁 Datasets
The pre-extracted multimodal features (text, audio, visual) used in this project are adopted from the [M3Net](https://github.com/feiyuchen7/M3NET) project(Chen et al., CVPR 2023) and [GraphSmile](https://github.com/lijfrank/GraphSmile) project(Li, Wang, and Zeng, 2025). Download multimodal features:
- [IEMOCAP](https://drive.google.com/drive/folders/1s5S1Ku679nlVZQPEfq-6LXgoN1K6Tzmz?usp=drive_link) → Place into the `IEMOCAP_features/` folder  
- [MELD](https://drive.google.com/drive/folders/1GfqY7WNVeCBWoFa_NSTalnaIgyyOVJuC?usp=drive_link) → Place into the `MELD_features/` folder
- [CMU-MOSEI](https://drive.google.com/drive/folders/1_j3w21zdYvA1yBajubXnaoIhYM22kI3P?usp=drive_link) → Place into the `CMU-MOSEI_features/` foler


Download raw datasets(Optional):
- [IEMOCAP](https://sail.usc.edu/iemocap/)
- [MELD](https://github.com/SenticNet/MELD)
- [CMU-MOSEI](http://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/)

## 🏋️‍♀️ Training

### Train on IEMOCAP
```bash
python -u train.py --base-model 'GRU' --dropout 0.5 --lr 0.0001 --batch-size=32 --epochs=60 --multi_modal --mm_fusion_mthd='concat_DHT' --modals='avl' --Dataset='IEMOCAP' --norm BN --num_L=6 --num_K=3 --window_spk=10 --window_spk_f=-1 --window_dir=1 --window_dir_f=-1 --epsilon2=1 --epsilon=1 --use_speaker='bh' --multimodal_node='both' --graph_type='both' --directed_edge='avl' --single_edge=''
```

### Train on MELD
```bash
python -u train.py --base-model 'GRU' --dropout=0.4 --lr=0.0001 --batch-size 32 --epochs=6 --multi_modal --mm_fusion_mthd='concat_DHT' --modals='avl' --Dataset='MELD' --norm BN --num_L=1 --num_K=1 --window_spk=3 --window_spk_f=1 --window_dir=8 --window_dir_f=6 --epsilon2=0.1 --epsilon=1.1 --use_speaker='i' --multimodal_node='both' --graph_type='both' --directed_edge='avl' --single_edge=''
```

### Train on CMU-MOSEI
```bash
python3 -u train_MOSEI.py --base-model GRU --dropout=0.4 --lr=0.0001 --batch-size 8 --epochs=30 --multi_modal --mm_fusion_mthd=concat_DHT --modals=avl --Dataset=MOSEI --norm=BN --num_L=1 --num_K=1 --window_spk=11 --window_spk_f=1 --window_dir=1 --window_dir_f=2 --epsilon=0.1 --epsilon2=0.9 --use_speaker='' --multimodal_node='both' --graph_type='both' --directed_edge='avl' --single_edge=''
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

### Inference with Pretrained Checkpoint on CMU-MOSEI
```bash
python3 -u train.py --base-model GRU --dropout=0.4 --lr=0.0001 --batch-size 8 --epochs=30 --multi_modal --mm_fusion_mthd=concat_DHT --modals=avl --Dataset=MOSEI --norm=BN --num_L=1 --num_K=1 --window_spk=11 --window_spk_f=1 --window_dir=1 --window_dir_f=2 --epsilon=0.1 --epsilon2=0.9 --use_speaker='' --multimodal_node='both' --graph_type='both' --directed_edge='avl' --single_edge=''
```
> Checkpoint path: `./best_model_MOSEI.pth`

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
# Table 3 (i) w/o multimodal node (h^m)
--multimodal_node=''
# Table 3 (ii) w/o EIGNN (I^m)
--graph_type='ECG'
# Table 3 (iii) w/o ECGNN (C^m)
--graph_type='EIG'
# Table 4 Modality Configuration
Remove --multi_modal
--modals='l' --direcred_edge='l' or --modals='a' --direcred_edge='a' or --modals='v' --direcred_edge='v'
Do not remove --multi_modal
--modals='la' --direcred_edge='la' or --modals='lv' --direcred_edge='lv' or --modals='av' --direcred_edge='av'
# Appendix Table 5 Effect of Speaker Embedding
--use_speaker=''
# Appendix Table 6 Effect of Intra-Utterance Multi-Edges
--single_edge='intra'
# Appendix Table 7 Effect of Intra-Utterance Directed Edges
--directed_edge='' or --directed_edge='l' or --directed_edge='a' or --directed_edge='v' or --directed_edge='la' or --directed_edge='lv' or --directed_edge='av'
```

### Effect of Hyperparameter
We also evaluate the model’s sensitivity to key hyperparameters:
```bash
# Figure 3: Number of GNN Layers
--num_L=<int>
--num_K=<int>
# Appendix Figure 3: Impact of Residual Scaling εI and εC
# For EIGNN
--epsilon2=<float>
# For ECGNN
--epsilon=<float>
# Appendix Figure 4: Effect of Window Size
# For EIGNN
--window_spk=<int>
--window_spk_f=<int>
# For ECGNN
--window_dir=<int>
--window_dir_f=<int>
```

### Additional Analyses after Train
```bash
# Figure 2: t-SNE visualization on IEMOCAP
python tsne_IEMOCAP.py
# Appendix Figure 5: t-SNE visualization on MELD
python tsne_MELD.py
# Table 5: Memory, and Inference Time Overhead
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
