import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report

df = pd.read_csv("saved_outputs/utterances_MELD.csv")
df = df.sort_values(by=["dialogue_id", "utterance_idx"]).reset_index(drop=True)

prev_utterances = {}
inertia_idxs = []
without_inertia_idxs = []
for i, row in df.iterrows():
    d_id = row["dialogue_id"]
    u_idx = row["utterance_idx"]
    speaker = row["speaker"]
    label = row["label"]

    if d_id not in prev_utterances:
        prev_utterances[d_id] = {}

    if speaker in prev_utterances[d_id]:
        prev_label, _ = prev_utterances[d_id][speaker]
        if label == prev_label:
            inertia_idxs.append(i)
        else:
            without_inertia_idxs.append(i)
    else:
        without_inertia_idxs.append(i)

    prev_utterances[d_id][speaker] = (label, i)
    

inertia_f1 = f1_score(df.loc[inertia_idxs, "label"], df.loc[inertia_idxs, "pred"], average='weighted') if inertia_idxs else 0
contagion_f1 = f1_score(df.loc[without_inertia_idxs, "label"], df.loc[without_inertia_idxs, "pred"], average='weighted') if without_inertia_idxs else 0

print(f"Emotional Inertia: {len(inertia_idxs)}, w-F1: {inertia_f1:.4f}")
print(f"w/o Emotional Inertia: {len(without_inertia_idxs)}, w-F1: {contagion_f1:.4f}")


best_mask = None
best_label = df.loc[inertia_idxs, "label"]
best_pred =  df.loc[inertia_idxs, "pred"]
print("Emotional Inertia")
print(classification_report(best_label,best_pred, sample_weight=best_mask,digits=4))
print(confusion_matrix(best_label,best_pred,sample_weight=best_mask))

best_label = df.loc[without_inertia_idxs, "label"]
best_pred =  df.loc[without_inertia_idxs, "pred"]
print("w/o Emotional Inertia")
print(classification_report(best_label,best_pred, sample_weight=best_mask,digits=4))
print(confusion_matrix(best_label,best_pred,sample_weight=best_mask))