# XSS Detector Project

A powerful XSS (Cross-Site Scripting) payload detection model built with PyTorch. The model uses a hybrid architecture combining multi-kernel CNN, attention mechanism, and bidirectional LSTM, trained with Focal Loss to handle class imbalance effectively.

## 📋 Project Overview

This project detects malicious XSS payloads using deep learning. It includes:

- Custom preprocessing and tokenization tailored for XSS payloads
- Vocabulary built from training data
- Hybrid model: CNN + Attention + BiLSTM
- Focal Loss with label smoothing
- AdamW optimizer + Cosine Annealing scheduler
- Early stopping and hard example mining
- Full training and inference scripts

## 📁 Project Structure
xss-detector-project/
├── data/                   # Place your datasets here
│   ├── XSS_dataset.csv     # Training dataset
│   └── xss_da.csv          # Test/New dataset
├── models/                 # Trained model will be saved here
├── src/
│   ├── config.py           # Hyperparameters
│   ├── preprocess.py       # Preprocessing & tokenization
│   ├── model.py            # Model architecture & Focal Loss
│   ├── train.py            # Training script
│   └── inference.py        # Evaluation on new data
├── requirements.txt
├── README.md
└── .gitignore


## 🚀 Quick Start

### Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/xss-detector.git
cd xss-detector