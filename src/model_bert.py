from transformers import BertTokenizer, BertForSequenceClassification

from .config import BERT_MODEL, BERT_MAX_LEN

def get_bert_model_and_tokenizer(vocab_size=None):
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
    model = BertForSequenceClassification.from_pretrained(BERT_MODEL, num_labels=2)
    return model, tokenizer