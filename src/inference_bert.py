import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset as TorchDataset
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix,
    average_precision_score, precision_recall_fscore_support
)

from .config import BERT_MODEL_NAME, BERT_MAX_LEN, BATCH_INF

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class InferenceDataset(TorchDataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], padding='max_length', truncation=True, max_length=BERT_MAX_LEN, return_tensors='pt')
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

df_new = pd.read_csv('data/xss_da.csv').dropna(subset=['sentence', 'Label'])
texts = df_new['sentence'].astype(str).values
labels = df_new['Label'].astype(int).values
print(f'new datasets: {len(texts)} sample')

tokenizer = BertTokenizer.from_pretrained(f'models/{BERT_MODEL_NAME}')
model = BertForSequenceClassification.from_pretrained(f'models/{BERT_MODEL_NAME}').to(DEVICE)
model.eval()

dataset = InferenceDataset(texts, labels, tokenizer)
loader = DataLoader(dataset, batch_size=BATCH_INF, shuffle=False)

probs = []
with torch.no_grad():
    for batch in loader:
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        outputs = model(input_ids, attention_mask=attention_mask)
        probs_batch = torch.softmax(outputs.logits, dim=1)[:,1].cpu().numpy()
        probs.extend(probs_batch)

probs = np.array(probs)
y_true = labels
y_pred = (probs > 0.7).astype(int)

print('\n========== bert result in new dataset ==========\n ==========')
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