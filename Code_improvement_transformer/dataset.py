import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
from typing import Any

class CodeDataset(Dataset):
    def __init__(self, tokenizer_src, tokenizer_tgt, json_file):
        super().__init__()
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.dataset = pd.read_json(json_file)
        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")])
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")])
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        src_code = self.dataset.iloc[idx, 0]  
        tgt_code = self.dataset.iloc[idx, 1]  
        src_tokens = self.tokenizer_src.encode(src_code)
        tgt_tokens = self.tokenizer_tgt.encode(tgt_code)
        if src_tokens[-1] != self.tokenizer_src.token_to_id("[EOS]"):
            src_tokens.append(self.tokenizer_src.token_to_id("[EOS]"))
        encoder_input = torch.cat(
            [
                self.sos_token, 
                torch.tensor(src_tokens), 
                self.eos_token,
                torch.tensor([self.tokenizer_src.token_to_id("[PAD]")]* (self.tokenizer_src.get_vocab_size() - len(src_tokens)))   
            ]
        )
        decoder_input = torch.cat(
            [
                self.sos_token, 
                torch.tensor(tgt_tokens), 
                self.eos_token,
                torch.tensor([self.tokenizer_tgt.token_to_id("[PAD]")]* (self.tokenizer_tgt.get_vocab_size() - len(tgt_tokens)))   
            ]
        )
        label = torch.cat(
            [
                torch.tensor(tgt_tokens), 
                self.eos_token,
                torch.tensor([self.tokenizer_tgt.token_to_id("[PAD]")]* (self.tokenizer_tgt.get_vocab_size() - len(tgt_tokens)))   
            ]
        )   

        assert encoder_input.size(0) == self.seq.len
        assert decoder_input.size(0) == self.seq.len
        assert label.size(0) == self.seq.len
        return{
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(-2),
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(-2),
            "label": label,
            "src_code": src_code,
            "tgt_code": tgt_code
        }
def casual_mask(size):
    mask = (torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int))

    return mask == 0
    
  
