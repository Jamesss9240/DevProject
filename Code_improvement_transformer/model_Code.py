import torch
import torch.nn as nn
import math
import re
from torch.nn import TransformerEncoderLayer, TransformerDecoderLayer, TransformerEncoder, TransformerDecoder

#reference used to help build base of transformer model: https://www.youtube.com/watch?v=ISNdQcPhsts

def generate_square_subsequent_mask(sz):

    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

class SafeCrossEntropyLoss(nn.Module):
    
    def __init__(self, ignore_index=-100, vocab_size=None):
        super(SafeCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.vocab_size = vocab_size
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')
    
    def forward(self, logits, targets):
        batch_size = targets.shape[0] if len(targets.shape) > 0 else 1
        
        #create mask for targets
        valid_mask = (targets != self.ignore_index)
        if torch.any(valid_mask):
            #check if targets in bounds
            max_target = targets.max().item() if targets.numel() > 0 else 0
            if max_target >= self.vocab_size:
                #clone+clamp targets, if out of bounds
                safe_targets = targets.clone()
                safe_targets[valid_mask] = torch.clamp(safe_targets[valid_mask], 0, self.vocab_size - 1)
                targets = safe_targets
        
        
        loss = self.criterion(logits, targets)
        
     
        valid_count = valid_mask.sum().item()
        if valid_count > 0:
            return loss.sum() / valid_count
        else:
            return loss.sum()  



class SimpleCodeTokenizer:
    def __init__(self):
        # Special tokens
        self.special_tokens = ["<pad>", "<unk>", "<eos>", "<bos>", "<newline>", "<indent>", "<dedent>"]
        self.vocab = {tok: idx for idx, tok in enumerate(self.special_tokens)}
        # Add all 256 possible byte values
        for i in range(256):
            self.vocab[f"<byte_{i}>"] = len(self.vocab)
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)

    def encode(self, text, max_length=128, padding="max_length", truncation=True, return_tensors=None):
        tokens = [self.vocab["<bos>"]]
        lines = text.splitlines(keepends=True)
        indent_stack = [0]
        for line in lines:
            # Count leading spaces/tabs for indentation
            stripped = line.lstrip('\t ')
            indent = len(line) - len(stripped)
            if indent > indent_stack[-1]:
                tokens.append(self.vocab["<indent>"])
                indent_stack.append(indent)
            while indent < indent_stack[-1]:
                tokens.append(self.vocab["<dedent>"])
                indent_stack.pop()
            # Encode line content
            for b in stripped.encode("utf-8", errors="replace"):
                tokens.append(self.vocab[f"<byte_{b}>"])
            if line.endswith('\n'):
                tokens.append(self.vocab["<newline>"])
        # Close any remaining indents
        while len(indent_stack) > 1:
            tokens.append(self.vocab["<dedent>"])
            indent_stack.pop()
        tokens.append(self.vocab["<eos>"])
        if truncation and len(tokens) > max_length:
            tokens = tokens[:max_length-1] + [self.vocab["<eos>"]]
        attention_mask = [1] * len(tokens)
        if padding == "max_length":
            pad_len = max_length - len(tokens)
            if pad_len > 0:
                tokens += [self.vocab["<pad>"]] * pad_len
                attention_mask += [0] * pad_len
        if return_tensors == "pt":
            import torch
            return {
                "input_ids": torch.tensor([tokens], dtype=torch.long),
                "attention_mask": torch.tensor([attention_mask], dtype=torch.long)
            }
        else:
            return {
                "input_ids": [tokens],
                "attention_mask": [attention_mask]
            }

    def decode(self, tokens, skip_special_tokens=True):
        if isinstance(tokens, (list, tuple)) and len(tokens) == 1 and isinstance(tokens[0], (list, torch.Tensor)):
            tokens = tokens[0]
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        bytes_out = bytearray()
        result = ""
        indent_level = 0
        for token_id in tokens:
            token_str = self.id_to_token.get(token_id, "")
            if token_str == "<newline>":
                result += "\n" + (" " * indent_level)
            elif token_str == "<indent>":
                indent_level += 4  # or 1 tab, adjust as needed
            elif token_str == "<dedent>":
                indent_level = max(0, indent_level - 4)
            elif token_str in self.special_tokens:
                if skip_special_tokens:
                    continue
                else:
                    result += token_str
            elif token_str.startswith("<byte_") and token_str.endswith(">"):
                try:
                    byte_val = int(token_str[6:-1])
                    result += bytearray([byte_val]).decode("utf-8", errors="replace")
                except Exception:
                    pass
        return result


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask



class SimpleCodeModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        pad_token_id=0,
        bos_token_id=3,
        eos_token_id=2,
        d_model=1024,
        nhead=16,
        num_encoder_layers=8,
        num_decoder_layers=8,
        dim_feedforward=4096,
        dropout=0.25,
        
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.layernorm = nn.LayerNorm(d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Linear(d_model, vocab_size)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.fc.weight, gain=0.01)
        nn.init.zeros_(self.fc.bias)

    def forward(self, src, tgt):
        src = torch.clamp(src, 0, self.vocab_size - 1)
        tgt = torch.clamp(tgt, 0, self.vocab_size - 1)

        src_emb = self.embedding(src)
        src_emb = self.positional_encoding(src_emb)
        src_emb = self.layernorm(src_emb)

        tgt_emb = self.embedding(tgt)
        tgt_emb = self.positional_encoding(tgt_emb)
        tgt_emb = self.layernorm(tgt_emb)

        src_key_padding_mask = (src == self.pad_token_id)
        tgt_key_padding_mask = (tgt == self.pad_token_id)
        tgt_mask = generate_square_subsequent_mask(tgt.size(1)).to(src.device)

        output = self.transformer(
            src_emb, tgt_emb,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            tgt_mask=tgt_mask
        )
        logits = self.fc(output)
        return logits

    def generate(self,input_ids,max_length=256,temperature=0.7,do_sample=True,top_k=50,min_output_length=None,**kwargs):
        device = input_ids.device
        batch_size = input_ids.shape[0]
        self.eval()
        with torch.no_grad():
            #encode input
            src_emb = self.embedding(input_ids) * math.sqrt(self.d_model)
            src_emb = self.positional_encoding(src_emb)
            memory = self.transformer.encoder(src_emb)
            #start generation using <bos> token
            current_output = torch.full((batch_size, 1), self.bos_token_id, dtype=torch.long, device=device)
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
            if min_output_length is None:
                min_output_length = input_ids.size(1)
            for step in range(max_length - 1):
                #prepare decoder input
                tgt_mask = generate_square_subsequent_mask(current_output.size(1)).to(device)
                tgt_emb = self.embedding(current_output) * math.sqrt(self.d_model)
                tgt_emb = self.positional_encoding(tgt_emb)
                output = self.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
                logits = self.fc(output[:, -1])
                logits = logits / temperature
                if do_sample:
                    if top_k > 0:
                        top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
                        logits_new = torch.full_like(logits, float('-inf'))
                        logits_new.scatter_(1, top_k_indices, top_k_logits)
                        logits = logits_new
                        #sample next tokens
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    #greedy decoding
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                #prevent stopping before minimum output length is reached, model likes to shorten inputs so thi helps stop that
                for b in range(batch_size):
                    if current_output.size(1) < min_output_length and next_token[b].item() == self.eos_token_id:
                        logits[b, self.eos_token_id] = float('-inf')
                        if do_sample:
                            probs = torch.softmax(logits[b], dim=-1)
                            next_token[b] = torch.multinomial(probs, num_samples=1)
                        else:
                            next_token[b] = torch.argmax(logits[b], dim=-1, keepdim=True)
                current_output = torch.cat([current_output, next_token], dim=1)             
                finished = finished | ((next_token.squeeze(-1) == self.eos_token_id) & (current_output.size(1) >= min_output_length))
    #generation loop done, process outputs
            outputs = []
            for b in range(batch_size):
                out = current_output[b].tolist()
                if out and out[0] == self.bos_token_id:
                    out = out[1:]              
                if self.eos_token_id in out:
                    out = out[:out.index(self.eos_token_id)]             
                while out and out[-1] == self.pad_token_id:
                    out.pop()
                outputs.append(torch.tensor(out, dtype=torch.long, device=device))
            max_len = max(len(o) for o in outputs)
            padded = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long, device=device)
            for i, o in enumerate(outputs):
                padded[i, :len(o)] = o
            return padded
    def generate_with_metrics(
        self,
        metrics_head,
        tokenizer,
        code,
        language,
        device=None,
        max_length=256,
        temperature=0.7,
        do_sample=True,
        top_k=50,
        top_p=0.95
    ):
        
        if device is None:
            device = next(self.parameters()).device

        self.eval()
        metrics_head.eval()

        # Encode the input code for metrics
        encoded = tokenizer.encode(
            f"### LANGUAGE: {language.upper()}\n### CODE QUALITY:\n{code}",
            return_tensors="pt"
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        # Get code representation from encoder
        with torch.no_grad():
            src_emb = self.embedding(input_ids) * math.sqrt(self.d_model)
            src_emb = self.positional_encoding(src_emb)
            src_emb = self.layernorm(src_emb)
            encoder_output = self.transformer.encoder(src_emb, src_key_padding_mask=(input_ids == self.pad_token_id))
            code_repr = encoder_output.mean(dim=1)
            predicted_metrics = metrics_head(code_repr)

        # Generate improved code
        improved_code_input = tokenizer.encode(
            f"### LANGUAGE: {language.upper()}\n### BUGGY CODE:\n{code}\n### FIXED CODE:\n",
            return_tensors="pt"
        )
        improved_input_ids = improved_code_input["input_ids"].to(device)
        improved_outputs = self.generate(
            improved_input_ids,
            max_length=max_length,
            temperature=temperature,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p
        )
        improved_code = tokenizer.decode(improved_outputs[0], skip_special_tokens=True)
        if "### FIXED CODE:" in improved_code:
            improved_code = improved_code.split("### FIXED CODE:", 1)[-1].strip()

        # Convert metrics to dictionary
        metrics_dict = {
            "security": predicted_metrics[0][0].item(),
            "maintainability": predicted_metrics[0][1].item(),
            "performance": predicted_metrics[0][2].item(),
            "reliability": predicted_metrics[0][3].item(),
            "fucntionality": predicted_metrics[0][4].item(),
        }

        # save metrics to file
        try:
            with open("generated_metrics_log.txt", "a", encoding="utf-8") as f:
                f.write(f"LANGUAGE: {language}\nCODE:\n{code}\nMETRICS: {metrics_dict}\n\n")
        except Exception as e:
            print(f"Could not save metrics log: {e}")

        return {
            "original_code": code,
            "improved_code": improved_code,
            "metrics": metrics_dict
        }

  
       
class LanguageClassifierHead(nn.Module):
        def __init__(self, base_model, num_classes):
            super().__init__()
            self.base_model = base_model
            self.dropout = nn.Dropout(0.3)  
            self.classifier = nn.Linear(1024, num_classes)
            nn.init.zeros_(self.classifier.bias)
            nn.init.xavier_uniform_(self.classifier.weight)

        def forward(self, input_ids, attention_mask=None):
            try:
                embedded = self.base_model.encoder_embedding(input_ids) * math.sqrt(self.base_model.d_model)
                embedded = embedded.transpose(0, 1)  
                embedded = self.base_model.positional_encoding(embedded)
                if attention_mask is not None:
                    padding_mask = ~attention_mask.bool().transpose(0, 1)
                else:
                    padding_mask = None
                
                encoder_outputs = self.base_model.transformer_encoder(
                    embedded, src_key_padding_mask=padding_mask
                )
                output = encoder_outputs.mean(dim=0)
                output = self.dropout(output)
                
                return self.classifier(output)
            except Exception as e:
                print(f"Error in forward method")
                


class LanguageDiscriminator(nn.Module):
    def __init__(self, hidden_size, num_languages, dropout=0.1):  
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)  
        self.classifier = nn.Linear(hidden_size, num_languages)
        
    def forward(self, hidden_states):
        x = self.dense(hidden_states)
        x = self.activation(x)
        x = self.dropout(x)  
        return self.classifier(x)


class LabelSmoothedCrossEntropyLoss(nn.Module):
    def __init__(self, epsilon=0.1, ignore_index=-100):
        super(LabelSmoothedCrossEntropyLoss, self).__init__()
        self.epsilon = epsilon
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        vocab_size = logits.size(-1)
        

        one_hot_targets = torch.zeros_like(logits)
        valid_mask = (targets != self.ignore_index)
        valid_targets = targets.masked_fill(~valid_mask, 0)
        one_hot_targets.scatter_(1, valid_targets.unsqueeze(1), 1)
        smoothed_targets = one_hot_targets * (1.0 - self.epsilon) + self.epsilon / vocab_size
        log_probs = torch.log_softmax(logits, dim=1)
        loss = -torch.sum(smoothed_targets * log_probs, dim=1)
        
        loss = loss * valid_mask.float()
        return loss.sum() / valid_mask.sum().clamp(min=1.0)

def generate_square_subsequent_mask(sz):

    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  
        self.register_buffer('pe', pe)

    def forward(self, x):
      
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)




def generate_square_subsequent_mask(sz):

    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def build_transformer_debugger(
    src_vocab_size, tgt_vocab_size, num_classes,
    d_model=1024, heads=16, 
    num_encoder_layers=8, num_decoder_layers=8,
    d_ff=4096, dropout=0.25, device="cpu"
):
 
    return SimpleCodeModel(
        vocab_size=src_vocab_size,
        d_model=d_model,
        nhead=heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=d_ff,
        dropout=dropout,
    ).to(device)