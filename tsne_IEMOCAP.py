import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE

save_dir = "saved_features"
label_names = ["Happy", "Sad", "Neutral", "Anger", "Excited", "Frustrated"]

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
        "Happy": "#ff7f0e",
        "Sad": "#9467bd",
        "Neutral": "#1f77b4",
        "Anger": "#2ca02c",
        "Excited": "#8c564b",
        "Frustrated": "#d62728"
    }

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="x", y="y", hue="label", palette=palette, alpha=0.7, s=80)
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

    speaker_ids = sorted(set(speakers))

    custom_palette = {
        "Speaker 1": "#999999",
        "Speaker 2": "#66c2a5"
    }
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="x", y="y", hue="speaker", palette=custom_palette, alpha=1.0, s=80)
    plt.xlabel("")
    plt.ylabel("")
    plt.tick_params(labelsize=18)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved: {filename}")

emotion_before = to_numpy(load_pickle("multimodal_emotions_before_IEMOCAP.pkl"))
emotion_after = to_numpy(load_pickle("multimodal_emotions_after_IEMOCAP.pkl"))
labels = np.array(load_pickle("emotion_labels_IEMOCAP.pkl"))

spk_idx = to_numpy(load_pickle("speaker_index_IEMOCAP.pkl"))
spk_named = np.array([f"Speaker {i+1}" for i in spk_idx])

reduced_before = reduce_with_tsne(emotion_before, seed=42)
reduced_after = reduce_with_tsne(emotion_after, seed=42)


plot_with_emotion_labels(reduced_before, labels, "IEMOCAP_emotion_only_tsne_before.png")
plot_with_emotion_labels(reduced_after,  labels, "IEMOCAP_emotion_only_tsne_after.png")

plot_with_speaker_labels(reduced_before, spk_named, "IEMOCAP_speaker_only_tsne_before.png")
plot_with_speaker_labels(reduced_after,  spk_named, "IEMOCAP_speaker_only_tsne_after.png")
