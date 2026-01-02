import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix,
    average_precision_score, precision_recall_fscore_support
)

from .config import MODEL_NAME, MAX_LEN, BATCH_INF
from .preprocess import xss_preprocess, xss_tokenize, encode
from .model import XSSDetector

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df_new = pd.read_csv('data/xss_da.csv').dropna(subset=['sentence', 'Label'])
payloads = df_new['sentence'].astype(str).values
labels = df_new['Label'].astype(int).values
print(f'new dataset: {len(payloads)} sample')

ckpt = torch.load(f'models/{MODEL_NAME}', map_location=DEVICE)
word2idx = ckpt['word2idx']
vocab_size = len(word2idx)

model = XSSDetector(vocab_size).to(DEVICE)
model.load_state_dict(ckpt['model'])
model.eval()

tok_new = [xss_tokenize(xss_preprocess(p)) for p in payloads]
X_new = encode(tok_new, word2idx)
y_new = torch.LongTensor(labels)

test_loader = DataLoader(TensorDataset(X_new, y_new), batch_size=BATCH_INF, shuffle=False, pin_memory=False)

probs_list = []
with torch.no_grad():
    for xb, _ in test_loader:
        xb = xb.to(DEVICE)
        pb = torch.softmax(model(xb), dim=1)[:, 1].cpu()
        probs_list.append(pb)

probs = torch.cat(probs_list).numpy()
y_true = labels
y_pred = (probs > 0.5).astype(int)

print('\n========== results in new dataset ==========')
print(f'AUC               : {roc_auc_score(y_true, probs):.4f}')
print(f'Accuracy          : {accuracy_score(y_true, y_pred):.4f}')
print(f'Precision (macro) : {precision_score(y_true, y_pred, average="macro"):.4f}')
print(f'Recall (macro)    : {recall_score(y_true, y_pred, average="macro"):.4f}')
print(f'F1 (macro)        : {f1_score(y_true, y_pred, average="macro"):.4f}')
print(f'PR-AUC            : {average_precision_score(y_true, probs):.4f}')

prec, rec, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=[0,1])
df_cls = pd.DataFrame({'Class': [0, 1], 'Precision': prec, 'Recall': rec, 'F1': f1, 'Support': sup})
print('\nPer-class:')
print(df_cls.to_string(index=False))

print('\nConfusion Matrix:')
print(confusion_matrix(y_true, y_pred))