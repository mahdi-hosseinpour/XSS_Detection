import re
import html
import urllib.parse
import pandas as pd
import torch
from .config import MAX_LEN

def xss_preprocess(text: str) -> str:
    if pd.isna(text): return ''
    s = str(text).strip()
    s = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]+', ' ', s)
    s = urllib.parse.unquote(s)
    s = html.unescape(s)
    s = re.sub(r'\\u([0-9a-fA-F]{4})', lambda m: chr(int(m.group(1), 16)), s)
    s = s.replace('<', ' < ').replace('>', ' > ')
    s = re.sub(r'\s+', ' ', s).strip().lower()
    return s

def xss_tokenize(text: str):
    return re.findall(r"\w+|</?\w+|[<>\"'=;:/.\-]|\S", text)[:MAX_LEN]

def encode(tokens_list, word2idx, max_len=MAX_LEN):
    seqs = []
    for toks in tokens_list:
        seq = [word2idx.get(t, 1) for t in toks][:max_len]  # 1 = <UNK>
        seq += [0] * (max_len - len(seq))  # 0 = <PAD>
        seqs.append(seq)
    return torch.LongTensor(seqs)