import time
import numpy as np
import torch
from torch.utils.data import Dataset

from transformers import BertTokenizerFast, BertForMaskedLM
from mlm_pytorch import MLM

from process_dataset import read_format


class LanguageDataset(Dataset):
    def __init__(self, path):
        self.documents, self.embeddings = read_format(path)

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, idx):
        return self.documents[idx], self.embeddings[idx], tokenizer


tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
transformer = BertForMaskedLM.from_pretrained('bert-base-uncased')

start_time = time.time()

text = ['asfsafasf' for i in range(122)]

inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')

print(time.time() - start_time)
print(inputs)
'''
loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
'''
print(inputs.input_ids.shape)
rand = torch.rand(inputs.input_ids.shape)

trainer = MLM(
    transformer,
    mask_token_id=103,
    pad_token_id=0,
    mask_prob=0.15,
    replace_prob=0.90,
    mask_ignore_token_ids = [101, 102]
)#.cuda()
