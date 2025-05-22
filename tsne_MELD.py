import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

# plt.rcParams['font.family'] = 'Times New Roman'

save_dir = "saved_features"
label_names = ["Neutral", "Surprise", "Fear", "Sadness", "Joy", "Disgust", "Anger"]  

def load_pickle(name):
    with open(os.path.join(save_dir, name), "rb") as f:
        return pickle.load(f)

def to_numpy(data):
    if isinstance(data, list):
        data = [x.cpu().detach().numpy() if hasattr(x, 'cpu') else x for x in data]
        return np.concatenate(data, axis=0)
    return data.cpu().detach().numpy() if hasattr(data, 'cpu') else data

def reduce_with_tsne(data, seed=None):
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=5000, random_state=seed)
    return tsne.fit_transform(data)

def plot_with_emotion_labels(data, labels, filename):
    df = pd.DataFrame(data, columns=["x", "y"])
    df["label"] = [label_names[i] for i in labels]

    palette = {
        "Neutral":  "#1f77b4",
        "Surprise": "#ff7f0e",
        "Fear":     '#4b0082',
        "Sadness":  "#9467bd",
        "Joy":      "#2ca02c",
        "Disgust":  "#17becf",
        "Anger":    "#d62728"
    }

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="x", y="y", hue="label", palette=palette, alpha=0.7, s=50)
    plt.xlabel("")
    plt.ylabel("")
    plt.tick_params(labelsize=18)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved: {filename}")

def plot_with_speaker_labels(data, speakers, filename):
    df = pd.DataFrame(data, columns=["x", "y"])
    df["speaker"] = speakers
    custom_palette = {
        "Speaker 1": "#999999",
        "Speaker 2": "#66c2a5",
        "Speaker 3": "#fc8d62",
        "Speaker 4": "#8da0cb",
        "Speaker 5": "#e78ac3",
        "Speaker 6": "#a6d854",
        "Speaker 7": "#ffd92f",
        "Speaker 8": "#e5c494",
        "Speaker 9": "#b3b3b3",
        "Speaker 10": "#ffb3b3"
    }
    speaker_ids = sorted(set(speakers))

    score = silhouette_score(data, speakers)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="x", y="y", hue="speaker", palette=custom_palette, alpha=1.0, s=50)
    plt.xlabel("")
    plt.ylabel("")
    plt.tick_params(labelsize=18)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved: {filename}")

emotion_before = to_numpy(load_pickle("multimodal_emotions_before_MELD.pkl"))
emotion_after = to_numpy(load_pickle("multimodal_emotions_after_MELD.pkl"))  
labels = np.array(load_pickle("emotion_labels_MELD.pkl"))  
spk_idx = to_numpy(load_pickle("speaker_index_MELD.pkl"))
spk_named = np.array([f"Speaker {i+1}" for i in spk_idx])


reduced_before = reduce_with_tsne(emotion_before, seed=42)
reduced_after = reduce_with_tsne(emotion_after, seed=42)

plot_with_emotion_labels(reduced_before, labels, "MELD_emotion_only_tsne_before.png")
plot_with_emotion_labels(reduced_after,  labels, "MELD_emotion_only_tsne_after.png")

plot_with_speaker_labels(reduced_before, spk_named, "MELD_speaker_only_tsne_before.png")
plot_with_speaker_labels(reduced_after,  spk_named, "MELD_speaker_only_tsne_after.png")
