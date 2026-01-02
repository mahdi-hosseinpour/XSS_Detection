import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import Dataset

from .config import *
from .model_bert import get_bert_model_and_tokenizer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv('data/XSS_dataset.csv')
texts = df['sentence'].astype(str).values
labels = df['Label'].astype(int).values

train_txt, tmp_txt, y_train, y_tmp = train_test_split(texts, labels, test_size=0.4, stratify=labels, random_state=SEED)
val_txt, test_txt, y_val, y_test = train_test_split(tmp_txt, y_tmp, test_size=0.5, stratify=y_tmp, random_state=SEED)

model, tokenizer = get_bert_model_and_tokenizer()

train_data = {'text': train_txt, 'label': y_train}
val_data = {'text': val_txt, 'label': y_val}

train_dataset = Dataset.from_dict(train_data)
val_dataset = Dataset.from_dict(val_data)

def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=BERT_MAX_LEN)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

training_args = TrainingArguments(
    output_dir='models/bert_checkpoints',
    num_train_epochs=BERT_EPOCHS,
    per_device_train_batch_size=BERT_BATCH,
    per_device_eval_batch_size=BERT_BATCH,
    learning_rate=BERT_LR,
    weight_decay=WEIGHT_DECAY,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    Pythonclass_weight = True,
    metric_for_best_model='eval_loss',
    greater_is_better=False,
    seed=SEED,
    fp16=torch.cuda.is_available(),
    logging_strategy='epoch',
    report_to="none"
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=1)[:,1].numpy()
    auc = roc_auc_score(labels, probs)
    return {'auc': auc}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)]
)

trainer.train()

trainer.save_model(f'models/{BERT_MODEL_NAME}')
tokenizer.save_pretrained(f'models/{BERT_MODEL_NAME}')

print('Finished bert learning model')