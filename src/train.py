import torch
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

from .config import *
from .preprocess import xss_preprocess, xss_tokenize, encode
from .model import XSSDetector, FocalLoss

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv('data/XSS_dataset.csv')
texts = df['sentence'].astype(str).values
labels = df['Label'].astype(int).values

train_txt, tmp_txt, y_train, y_tmp = train_test_split(texts, labels, test_size=0.4, stratify=labels, random_state=SEED)
val_txt, test_txt, y_val, y_test = train_test_split(tmp_txt, y_tmp, test_size=0.5, stratify=y_tmp, random_state=SEED)

train_tok = [xss_tokenize(xss_preprocess(t)) for t in train_txt]
val_tok = [xss_tokenize(xss_preprocess(t)) for t in val_txt]

counter = Counter(tok for sent in train_tok for tok in sent)
vocab = ['<PAD>', '<UNK>'] + [w for w, _ in counter.most_common(MAX_VOCAB - 2)]
word2idx = {w: i for i, w in enumerate(vocab)}
vocab_size = len(vocab)

X_train = encode(train_tok, word2idx)
X_val = encode(val_tok, word2idx)
y_train = torch.LongTensor(y_train)
y_val = torch.LongTensor(y_val)
y_test = torch.LongTensor(y_test)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH, shuffle=True, drop_last=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH * 2)

net = XSSDetector(vocab_size).to(DEVICE)
criterion = FocalLoss()
optimizer = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_auc, patience = 0, 0
for epoch in range(1, EPOCHS + 1):
    net.train()
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(net(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        optimizer.step()
    scheduler.step()

    net.eval()
    with torch.no_grad():
        probs = torch.softmax(net(X_val.to(DEVICE)), dim=1)[:, 1].cpu().numpy()
    auc = roc_auc_score(y_val.numpy(), probs)
    print(f'Ep {epoch:02d}  Val-AUC={auc:.4f}')

    if auc > best_auc:
        best_auc = auc
        patience = 0
        torch.save({'model': net.state_dict(), 'word2idx': word2idx, 'max_len': MAX_LEN},
                   f'models/{MODEL_NAME}')
    else:
        patience += 1
        if patience >= PATIENCE:
            print('Early-stop')
            break

# Hard Example Mining
ckpt = torch.load(f'models/{MODEL_NAME}', map_location=DEVICE)
net.load_state_dict(ckpt['model'])
net.eval()

with torch.no_grad():
    probs_train = torch.softmax(net(X_train.to(DEVICE)), dim=1)[:, 1].cpu()
margin = torch.abs(probs_train - 0.5)
hard_idx = torch.where(margin < 0.3)[0]

if len(hard_idx) > 0:
    hard_set = TensorDataset(X_train[hard_idx], y_train[hard_idx])
    hard_loader = DataLoader(hard_set, batch_size=BATCH, shuffle=True)
    optimizer = torch.optim.AdamW(net.parameters(), lr=LR / 5)
    for _ in range(HARD_EPOCH):
        net.train()
        for x, y in hard_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(x), y)
            loss.backward()
            optimizer.step()

torch.save({'model': net.state_dict(), 'word2idx': word2idx, 'max_len': MAX_LEN},
           f'models/{MODEL_NAME}')
print('end training and save model')