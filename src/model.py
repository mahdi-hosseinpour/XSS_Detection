import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import ALPHA_FOCAL, GAMMA_FOCAL, SMOOTH

class FocalLoss(nn.Module):
    def __init__(self, alpha=ALPHA_FOCAL, gamma=GAMMA_FOCAL, smooth=SMOOTH):
        super().__init__()
        self.a, self.g, self.s = alpha, gamma, smooth

    def forward(self, logits, y):
        y_smooth = torch.zeros_like(logits).scatter_(1, y.view(-1,1), 1)
        y_smooth = y_smooth * (1-self.s) + self.s/2.
        log_p = F.log_softmax(logits, dim=1)
        p = torch.exp(log_p)
        loss = -y_smooth * torch.pow(1-p, self.g) * log_p
        alpha_t = torch.tensor([self.a, 1-self.a]).to(logits.device).view(1,2)
        loss = alpha_t * loss
        return loss.sum(dim=1).mean()

class XSSDetector(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.emb   = nn.Embedding(vocab_size, 128, padding_idx=0)
        self.drop1 = nn.Dropout(0.2)
        self.convs = nn.ModuleList([nn.Conv1d(128, 128, k, padding=k//2) for k in [3,5,7]])
        self.attn  = nn.Sequential(nn.Linear(128*3, 64), nn.Tanh(), nn.Linear(64, 1))
        self.lstm  = nn.LSTM(128*3, 256, batch_first=True, bidirectional=True)
        self.fc    = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.emb(x)
        x = self.drop1(x).permute(0,2,1)
        h = torch.cat([F.relu(c(x)) for c in self.convs], dim=1)
        h = h.permute(0,2,1)
        a = self.attn(h)
        a = torch.softmax(a, dim=1)
        h = h * a
        h, _ = self.lstm(h)
        h = torch.max(h, dim=1)[0]
        return self.fc(h)