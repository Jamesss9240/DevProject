import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import json
import random
import math
import numpy as np
from collections import Counter, defaultdict
import os
import re
import pickle
import traceback
from transformers import get_cosine_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import os 
import datetime
from difflib import SequenceMatcher
import string
import difflib
import random
from radon.complexity import cc_visit
from radon.metrics import mi_visit
from radon.raw import analyze

from model_Code import SimpleCodeModel, SimpleCodeTokenizer, generate_square_subsequent_mask, PositionalEncoding, LanguageClassifierHead


print("Running training")




IDENT_DATASET_PATH = r"C:\Users\James\Downloads\dev3\Code-improvement-web-app\Code_improvement_transformer\codedataset.json"
DEBUG_DATASET_PATH = r"C:\Users\James\Downloads\dev3\Code-improvement-web-app\Code_improvement_transformer\codedataset.json"
TRAINED_CLASSIFIER_PATH = "trained_language_classifier.pth"
FINAL_DEBUG_MODEL_PATH = "debug_model_final.pth"

    
def set_seed(seed=random.randint(0, 10000)):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(random.randint(0, 10000))




class BugFixDataset(torch.utils.data.Dataset):
    def __init__(self, examples, tokenizer):
        self.examples = []
        for ex in examples:
            #ensure that input/lkabel are tensors
            input_ids = ex['input_ids']
            labels = ex['labels']
            if not isinstance(input_ids, torch.Tensor):
                input_ids = torch.tensor(input_ids, dtype=torch.long)
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels, dtype=torch.long)
            #filter examples
            if tokenizer.decode(input_ids, skip_special_tokens=True).strip() and \
               tokenizer.decode(labels, skip_special_tokens=True).strip():
                self.examples.append({
                    'input_ids': input_ids,
                    'labels': labels,
                    'attention_mask': ex.get('attention_mask', torch.ones_like(input_ids)),
                    'language': ex.get('language', None),
                    'buggy_code': ex.get('buggy_code', tokenizer.decode(input_ids, skip_special_tokens=True)),
                    'fixed_code': ex.get('fixed_code', tokenizer.decode(labels, skip_special_tokens=True)),
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]



  

def extract_debug_examples(file_path):
    

    with open(file_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    debug_examples = []
    language_counts = Counter()
    
    for item in dataset:
        if "language" not in item or "example_code" not in item:
            continue
            
        language = item["language"]
        examples = item["example_code"]

        i = 0
        while i < len(examples) - 1:
            #error code
            if isinstance(examples[i], str) and (examples[i].startswith("// ERROR") or examples[i].startswith("# ERROR")):
                error_info = examples[i].split("ERROR:", 1)[1].strip() if "ERROR:" in examples[i] else ""
                error_lines = examples[i].split("\n", 1)
                buggy_code = error_lines[1] if len(error_lines) > 1 else ""
                #corrected code
                if (i+1) < len(examples) and (examples[i+1].startswith("// CORRECTED") or examples[i+1].startswith("# CORRECTED")):
                    corrected_lines = examples[i+1].split("\n", 1)
                    fixed_code = corrected_lines[1] if len(corrected_lines) > 1 else ""
                    
                    if buggy_code and fixed_code:
         
                        metrics = generate_quality_metrics(buggy_code, fixed_code, error_info)
                        
                        debug_examples.append({
                            "language": language,
                            "buggy_code": buggy_code,
                            "fixed_code": fixed_code,
                            "explanation": error_info,
                            "metrics": metrics  
                        })
                        language_counts[language] += 1
                i += 2
                continue       
            i += 1
    return debug_examples
    
    

def generate_quality_metrics(buggy_code, fixed_code, error_info):

    security_keywords = ["injection", "xss", "csrf", "auth", "password", "encrypt", "decrypt", "permission", "sql"]
    performance_keywords = ["loop", "complexity", "algorithm", "memory", "cpu", "timeout", "latency"]
    reliability_keywords = ["null", "exception", "error", "crash", "handle", "try", "catch", "validate"]
    maintainability_keywords = ["comment", "document", "naming", "indent", "format", "refactor"]
    

    base_security = random.randint(60, 90)  
    base_maintainability = random.randint(50, 95)
    base_performance = random.randint(65, 95)
    base_reliability = random.randint(55, 90)
    base_functionality = random.randint(70, 95)
    

    error_lower = error_info.lower()
    

    for keyword in security_keywords:
        if keyword in error_lower:
            base_security = max(10, base_security - random.randint(10, 30))
            break
    

    code_diff_ratio = abs(len(fixed_code) - len(buggy_code)) / (len(buggy_code) + 1)
    if code_diff_ratio > 0.5:  
        base_maintainability = max(10, base_maintainability - random.randint(10, 25))
    
    
    for keyword in performance_keywords:
        if keyword in error_lower:
            base_performance = max(10, base_performance - random.randint(15, 35))
            break
    
   
    for keyword in reliability_keywords:
        if keyword in error_lower:
            base_reliability = max(10, base_reliability - random.randint(15, 40))
            break
            
    
    severity_indicators = ["critical", "severe", "major", "crash", "failure"]
    for indicator in severity_indicators:
        if indicator in error_lower:
            base_functionality = max(10, base_functionality - random.randint(20, 50))
            break
            
    return {
        "security": base_security,
        "maintainability": base_maintainability,
        "performance": base_performance,
        "reliability": base_reliability,
        "functionality": base_functionality
    }

#cheate code example dataset
class DebugDataset(Dataset):
    def __init__(self, examples, code_tokenizer, ident_tokenizer=None, max_length=128):  
      
        self.examples = examples
        self.code_tokenizer = code_tokenizer
        self.ident_tokenizer = ident_tokenizer  
        self.max_length = max_length
        
        print(f"Created dataset with {len(self.examples)} code")
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        try:
            example = self.examples[idx]
            language = example["language"]
            buggy_code = example["buggy_code"]
            fixed_code = example["fixed_code"]
            explanation = example["explanation"] if "explanation" in example else ""
            
            # get metrics
            metrics = example.get("metrics", {
                "security": 75,
                "maintainability": 70, 
                "performance": 80,
                "reliability": 70,
                "functionality": 85
            })
            
            # language 
            input_text = f"fix {language} code: {buggy_code}"
            
            
            target_text = (
                f"fixed code: {fixed_code}\n\n"
                f"metrics: security={metrics['security']}, "
                f"maintainability={metrics['maintainability']}, "
                f"performance={metrics['performance']}, "
                f"reliability={metrics['reliability']}, "
                f"functionality={metrics['functionality']}\n\n"
                f"explanation: {explanation}"
            )
            
            #tokenizer encoding
            code_inputs = self.code_tokenizer.encode(
                input_text, 
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            code_targets = self.code_tokenizer.encode(
                target_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            result = {
                'input_ids': code_inputs['input_ids'].squeeze(0),
                'attention_mask': code_inputs['attention_mask'].squeeze(0),
                'labels': code_targets['input_ids'].squeeze(0),
            }
            
            # tokenize for identification model
            if self.ident_tokenizer is not None:
                try:
                    ident_encoding = self.ident_tokenizer.encode(
                        buggy_code, 
                        max_length=self.max_length,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt"
                    )
                    
                    result['ident_input_ids'] = ident_encoding['input_ids'].squeeze(0)
                    result['ident_attention_mask'] = ident_encoding['attention_mask'].squeeze(0)
                except Exception as e:
                    print(f"Error in id: {idx}: {e}")
                    result['ident_input_ids'] = torch.zeros(self.max_length, dtype=torch.long)
                    result['ident_attention_mask'] = torch.zeros(self.max_length, dtype=torch.long)
            
            return result
            
        except Exception as e:
            print(f"error {idx}: {e}")
 
            empty_tensor = torch.zeros(self.max_length, dtype=torch.long)
            result = {
                'input_ids': empty_tensor,
                'attention_mask': torch.zeros(self.max_length, dtype=torch.long),
                'labels': empty_tensor,
            }
            
            if self.ident_tokenizer is not None:
                result['ident_input_ids'] = empty_tensor
                result['ident_attention_mask'] = torch.zeros(self.max_length, dtype=torch.long)
                
            return result



class QualityMetricsHead(nn.Module):

    def __init__(self, input_dim=1024, hidden_dim=256):
        super(QualityMetricsHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 5)  
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  
        return x


class CodeQualityDataset(Dataset):

    def __init__(self, examples, tokenizer, max_length=256):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        code = example["code"]
        language = example["language"]
        metrics = example["metrics"]
        
       
        encoded = self.tokenizer.encode(
            f"### LANGUAGE: {language.upper()}\n### CODE QUALITY:\n{code}",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
 
        target_metrics = torch.tensor([
            metrics.get("security", 50),
            metrics.get("maintainability", 50),
            metrics.get("performance", 50),
            metrics.get("reliability", 50),
            metrics.get("functionality", 50)
        ], dtype=torch.float)
        
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "metrics": target_metrics,
            "language": language
        }

def train_metrics(code_model, code_tokenizer, bugfix_dataset, weighted_examples, 
                               languages, device, accuracy_target=0.85):
    print("\nphase 4 training metrics")
    
    metrics_categories = ['security', 'maintainability', 'performance', 'reliability', 'functionality']
    max_examples = min(4000, len(weighted_examples))
    selected_examples = random.sample(weighted_examples, max_examples)
    quality_examples = []
    
    for ex in selected_examples:
        try:
            buggy_code = ex["buggy_code"]
            fixed_code = ex["fixed_code"]
            language = ex["language"].lower()
            buggy_metrics = score_code_quality(buggy_code, language)
            fixed_metrics = score_code_quality(fixed_code, language)
            quality_examples.append({
                "code": buggy_code,
                "language": language,
                "quality": "low",
                "metrics": buggy_metrics
            })
            quality_examples.append({
                "code": fixed_code,
                "language": language,
                "quality": "high",
                "metrics": fixed_metrics
            })
        except Exception as e:
            print(f"Error generating metrics: {str(e)}")
    print(f"Created {len(quality_examples)}  examples")

    
    def normalize_metrics(metrics):
        return {k: float(metrics.get(k, 50)) for k in metrics_categories}

    for ex in quality_examples:
        ex["metrics"] = normalize_metrics(ex["metrics"])

    metrics_head = QualityMetricsHead(
        input_dim=code_model.d_model,
        hidden_dim=1024
    ).to(device)

    # freeze main model to not break it
    for param in code_model.parameters():
        param.requires_grad = False
    for param in metrics_head.parameters():
        param.requires_grad = True

    quality_dataset = CodeQualityDataset(quality_examples, code_tokenizer)
    train_indices, val_indices = train_test_split(
        range(len(quality_dataset)), test_size=0.1, random_state=42
    )
    batch_size = 8
    train_dataloader = DataLoader(
        torch.utils.data.Subset(quality_dataset, train_indices),
        batch_size=batch_size, shuffle=True, num_workers=1
    )
    val_dataloader = DataLoader(
        torch.utils.data.Subset(quality_dataset, val_indices),
        batch_size=batch_size, shuffle=False, num_workers=1
    )

    epochs = 30
    optimizer = optim.AdamW(metrics_head.parameters(), lr=2e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    patience = 7
    patience_counter = 0

    print(f"metrics trianing starting")

    for epoch in range(epochs):
        code_model.eval()  # Freeze main model
        metrics_head.train()
        epoch_loss = 0.0
        batch_count = 0

        for batch_idx, batch in enumerate(train_dataloader):
            try:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                target_metrics = batch['metrics'].to(device)

                with torch.no_grad():
                    src_emb = code_model.embedding(input_ids) * math.sqrt(code_model.d_model)
                    src_emb = code_model.positional_encoding(src_emb)
                    encoder_output = code_model.transformer.encoder(src_emb, src_key_padding_mask=~attention_mask.bool())
                code_repr = encoder_output.mean(dim=1)
                predicted_metrics = metrics_head(code_repr)
                loss = criterion(predicted_metrics, target_metrics)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(metrics_head.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
                batch_count += 1
                if batch_idx % 20 == 0:
                    print(f"Batch {batch_idx}/{len(train_dataloader)} - Loss: {loss.item():.4f}")
            except Exception as e:
                print(f"err in metrics batch {batch_idx}: {str(e)}")
                continue

        avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else float('inf')
        print(f"Epoch {epoch+1}/{epochs} - Avg loss: {avg_epoch_loss:.4f}")

        #validation
        code_model.eval()
        metrics_head.eval()
        val_loss = 0.0
        val_batch_count = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                try:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    target_metrics = batch['metrics'].to(device)
                    src_emb = code_model.embedding(input_ids) * math.sqrt(code_model.d_model)
                    src_emb = code_model.positional_encoding(src_emb)
                    encoder_output = code_model.transformer.encoder(src_emb, src_key_padding_mask=~attention_mask.bool())
                    code_repr = encoder_output.mean(dim=1)
                    predicted_metrics = metrics_head(code_repr)
                    loss = criterion(predicted_metrics, target_metrics)
                    val_loss += loss.item()
                    val_batch_count += 1
                except Exception as e:
                    print(f"error in validation batch {batch_idx}: {str(e)}")
                    continue

        avg_val_loss = val_loss / val_batch_count if val_batch_count > 0 else float('inf')
        scheduler.step(avg_val_loss)
        print(f"val_loss: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(metrics_head.state_dict(), "quality_metrics_head.pth")
            print("Saved metrics model")
            if avg_val_loss < (1 - accuracy_target):  # Lower loss means better fit
                print(f"eraly stopping")
                break
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"early stopping (no improvement)")
                break

    if os.path.exists("quality_metrics_head.pth"):
        metrics_head.load_state_dict(torch.load("quality_metrics_head.pth"))
    else:
        print("no metrics head found")

    print("metrics training complete.")
    return code_model, metrics_head
    






class TemplateDiscriminator(nn.Module):
   
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        return x
def custom_collate_fn(batch):
    batch_out = {}
    for key in batch[0]:
        #stack only if all tensors
        if all(isinstance(item[key], torch.Tensor) for item in batch):
            batch_out[key] = torch.stack([item[key] for item in batch])
        else:
            batch_out[key] = [item[key] for item in batch]
    return batch_out






def create_pure_preservation_examples(tokenizer, language, dataset_path, max_per_lang=200):

    examples = []
    pairs = []

    #get the code pairs from the dataset
    if dataset_path is not None:
        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                dataset = json.load(f)
            for entry in dataset:
                if entry.get("language", "").lower() == language.lower():
                    code_list = entry.get("example_code", [])
                    #make sure err/corrected in pairs
                    for i in range(0, len(code_list) - 1, 2):
                        buggy = code_list[i].replace("# ERROR:", "").replace("// ERROR:", "").strip()
                        fixed = code_list[i+1].replace("# CORRECTED", "").replace("// CORRECTED", "").strip()
                        pairs.append((fixed, fixed))  
                        if len(pairs) >= max_per_lang:
                            break
                if len(pairs) >= max_per_lang:
                    break
        except Exception as e:
            print(f"Error loading dataset {e}")

    
  
    pairs = pairs[:max_per_lang]
    for input_code, output_code in pairs:
        #encode the input and output code
        input_ids = tokenizer.encode(
            input_code,
            return_tensors="pt",
            max_length=128,
            padding="max_length",
            truncation=True
        )['input_ids'].squeeze(0)
        labels = tokenizer.encode(
            output_code,
            return_tensors="pt",
            max_length=128,
            padding="max_length",
            truncation=True
        )['input_ids'].squeeze(0)
        #decode the input and labels
        buggy_code = tokenizer.decode(input_ids, skip_special_tokens=True)
        fixed_code = tokenizer.decode(labels, skip_special_tokens=True)
        if buggy_code.strip() and fixed_code.strip():
            examples.append({
                'input_ids': input_ids,
                'attention_mask': (input_ids != tokenizer.vocab["<pad>"]).long(),
                'labels': labels,
                'language': language.lower(),
                'buggy_code': buggy_code,
                'fixed_code': fixed_code
            })
    print(f"Created {len(examples)} examples")
    return examples




def check_and_fix_embedding_size(model, tokenizer, device):
    if model.embedding.weight.size(0) != tokenizer.vocab_size:
        embed_dim = model.embedding.weight.size(1)
        new_embedding = nn.Embedding(
            num_embeddings=tokenizer.vocab_size,
            embedding_dim=embed_dim
        ).to(device)
        min_size = min(model.embedding.weight.size(0), tokenizer.vocab_size)
        with torch.no_grad():
            new_embedding.weight[:min_size] = model.embedding.weight[:min_size]
        model.embedding = new_embedding
        if hasattr(model, 'vocab_size'):
            model.vocab_size = tokenizer.vocab_size
    
    return model



class SafeDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, vocab_size):
        self.dataset = dataset
        self.vocab_size = vocab_size
        
    def __len__(self):
        return len(self.dataset)   

    def __getitem__(self, idx):
        item = self.dataset[idx]
        if isinstance(item, dict):
            if 'input_ids' in item:
                item['input_ids'] = torch.clamp(item['input_ids'], 0, self.vocab_size - 1)
            if 'labels' in item:
                safe_labels = item['labels'].clone()
                valid_label_mask = (safe_labels != -100)
                safe_labels[valid_label_mask] = torch.clamp(safe_labels[valid_label_mask], 0, self.vocab_size - 1)
                item['labels'] = safe_labels
            return item
        result = []
        for i, v in enumerate(item):
            if i == 0 and isinstance(v, torch.Tensor):  
                result.append(torch.clamp(v, 0, self.vocab_size - 1))
            elif i == 2 and isinstance(v, torch.Tensor):  
                safe_labels = v.clone()
                valid_label_mask = (safe_labels != -100)
                safe_labels[valid_label_mask] = torch.clamp(safe_labels[valid_label_mask], 0, self.vocab_size - 1)
                result.append(safe_labels)
            else:
                result.append(v)
        return tuple(result)



def filter_and_prepare_examples(examples, tokenizer):
    filtered = []
    for ex in examples:
        if not isinstance(ex['input_ids'], torch.Tensor):
            ex['input_ids'] = torch.tensor(ex['input_ids'])
        if not isinstance(ex['labels'], torch.Tensor):
            ex['labels'] = torch.tensor(ex['labels'])
        buggy = tokenizer.decode(ex['input_ids'], skip_special_tokens=True)
        fixed = tokenizer.decode(ex['labels'], skip_special_tokens=True)
        if buggy.strip() and fixed.strip():
            ex['buggy_code'] = buggy
            ex['fixed_code'] = fixed
            filtered.append(ex)
    return filtered

def ensure_tokenized_examples(examples, tokenizer, max_length=128):
    processed = []
    for ex in examples:
        if 'input_ids' in ex and 'labels' in ex:
            processed.append(ex)
            continue
        buggy_code = ex.get('buggy_code', '')
        fixed_code = ex.get('fixed_code', '')
        if not buggy_code or not fixed_code:
            continue 
        input_ids = tokenizer.encode(
            buggy_code,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )['input_ids'].squeeze(0)
        labels = tokenizer.encode(
            fixed_code,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )['input_ids'].squeeze(0)
        ex = ex.copy()
        ex['input_ids'] = input_ids
        ex['labels'] = labels
        processed.append(ex)
    return processed

def training_scheduler(model, tokenizer, train_dataloader, val_dataloader, device, epochs=60, patience=25, learning_rate=1e-5):
    print("\nStarting training\n")
    languages = ["python", "go", "c#"]
    best_model_paths = []

    #copy phase
    print("\nPHASE 0\n")
    phase0_examples = []
    language_counts = {}

    #get examples
    for language in languages:
        identity_examples = create_pure_preservation_examples(tokenizer, language, DEBUG_DATASET_PATH, max_per_lang=400)
        lang_examples = identity_examples
        phase0_examples.extend(lang_examples)
        language_counts[language] = len(lang_examples)


    #weigths to balance learning
    max_count = max(language_counts.values())
    language_weights = {lang: max_count / (count + 1) for lang, count in language_counts.items()}
    for ex in phase0_examples:
        lang = ex.get('language', '').lower()
        ex['weight'] = language_weights.get(lang, 1.0)

    random.shuffle(phase0_examples)
    phase0_examples = ensure_tokenized_examples(phase0_examples, tokenizer)
    phase0_examples = filter_and_prepare_examples(phase0_examples, tokenizer)

    class WeightedBugFixDataset(BugFixDataset):
        def __getitem__(self, idx):
            ex = super().__getitem__(idx)
            ex['weight'] = phase0_examples[idx].get('weight', 1.0)
            return ex

    identity_dataset = WeightedBugFixDataset(phase0_examples, tokenizer)
    safe_identity_dataset = SafeDatasetWrapper(identity_dataset, tokenizer.vocab_size)

    weights = np.array([ex['weight'] for ex in phase0_examples], dtype=np.float32)
    sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    identity_loader = DataLoader(
        safe_identity_dataset,
        batch_size=8,
        sampler=sampler,
        collate_fn=custom_collate_fn
    )
    #debugging
    for batch in identity_loader:
        print("PHASE 0 DEBUG BATCH INPUT IDs:", batch['input_ids'][0][:20].tolist())
        print("PHASE 0 DEBUG BATCH INPUT:", tokenizer.decode(batch['input_ids'][0], skip_special_tokens=True))
        print("PHASE 0 DEBUG BATCH LABEL:", tokenizer.decode(batch['labels'][0], skip_special_tokens=True))
        break
    from model_Code import SimpleCodeModel
    model = SimpleCodeModel(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.vocab["<pad>"],
        bos_token_id=tokenizer.vocab["<bos>"],
        eos_token_id=tokenizer.vocab["<eos>"],
        d_model=128,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=256,
        dropout=0.22,
    ).to(device)

    model = check_and_fix_embedding_size(model, tokenizer, device)
    model = train_phase0_identity_copy(model,tokenizer,identity_loader,device,epochs=60  )

    # save model+tokenizer
    torch.save(model.state_dict(), "debug_model_phase0_alllangs.pth")
    torch.save(model, "full0_alllangs.pt")
    with open("phase0_tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    print("copy phase complete.\n")
    torch.save(model, "phase0_multilang_model.pt")
    with open("phase0_tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

#bnugfix pair loading function
    def load_plain_bugfix_pairs(dataset_path, language):
        pairs = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for entry in data:
            if entry["language"].lower() == language.lower():
                codes = entry["example_code"]
                buggy = None
                fixed = None
                for line in codes:
                    
                    if re.match(r'^\s*(#|//)\s*ERROR', line, re.IGNORECASE):
                        buggy = line.split('\n', 1)[-1].strip() if '\n' in line else ''
                    elif re.match(r'^\s*(#|//)\s*CORRECTED', line, re.IGNORECASE):
                        fixed = line.split('\n', 1)[-1].strip() if '\n' in line else ''
                    else:
                        if buggy is not None and fixed is None:
                            buggy += '\n' + line
                        elif fixed is not None:
                            fixed += '\n' + line
                    if buggy is not None and fixed is not None:
                        pairs.append((buggy.strip(), fixed.strip()))
                        buggy = None
                        fixed = None
 
    print("\nsimple debug examples\n")
    dataset_path = r"C:\Users\James\Downloads\dev3\Code-improvement-web-app\Code_improvement_transformer\codedataset.json"
    minimal_examples = []
    for lang in languages:
        #add bugfix pairs
        for buggy, fixed in load_plain_bugfix_pairs(dataset_path, lang):
            input_ids = tokenizer.encode(buggy, max_length=128, padding="max_length", truncation=True, return_tensors="pt")['input_ids'].squeeze(0)
            output_ids = tokenizer.encode(fixed, max_length=128, padding="max_length", truncation=True, return_tensors="pt")['input_ids'].squeeze(0)
            minimal_examples.append({'input_ids': input_ids, 'labels': output_ids, 'language': lang, 'buggy_code': buggy, 'fixed_code': fixed})
    #add more examples and prepare
        minimal_examples.extend(create_pure_preservation_examples(tokenizer, lang, dataset_path, max_per_lang=40))
    minimal_examples = ensure_tokenized_examples(minimal_examples, tokenizer)
    minimal_examples = filter_and_prepare_examples(minimal_examples, tokenizer)
    minimal_dataset = BugFixDataset(minimal_examples, tokenizer)
    minimal_loader = DataLoader(SafeDatasetWrapper(minimal_dataset, tokenizer.vocab_size), batch_size=8, shuffle=True, collate_fn=custom_collate_fn)
    model = check_and_fix_embedding_size(model, tokenizer, device)
    model = train_phase1_minimal_bugfix(model, tokenizer, minimal_loader, device, epochs=40)
    torch.save(model.state_dict(), "debug_model_phase1.pth")
    torch.save(model, "phase1mdl.pt")
    with open("phase1_tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    #moderate bugfixing
    print("\nmoderate bugfixing\n")
    strict_examples = []
    for lang in languages:
        for buggy, fixed in load_plain_bugfix_pairs(dataset_path, lang):
            input_ids = tokenizer.encode(buggy, max_length=128, padding="max_length", truncation=True, return_tensors="pt")['input_ids'].squeeze(0)
            output_ids = tokenizer.encode(fixed, max_length=128, padding="max_length", truncation=True, return_tensors="pt")['input_ids'].squeeze(0)
            strict_examples.append({'input_ids': input_ids, 'labels': output_ids, 'language': lang, 'buggy_code': buggy, 'fixed_code': fixed})
        #add more examples
        strict_examples.extend(create_pure_preservation_examples(tokenizer, lang, dataset_path, max_per_lang=30))
    strict_examples = ensure_tokenized_examples(strict_examples, tokenizer)
    strict_examples = filter_and_prepare_examples(strict_examples, tokenizer)
    strict_dataset = BugFixDataset(strict_examples, tokenizer)
    strict_loader = DataLoader(SafeDatasetWrapper(strict_dataset, tokenizer.vocab_size), batch_size=8, shuffle=True, collate_fn=custom_collate_fn)
    
    model = check_and_fix_embedding_size(model, tokenizer, device)
    model = train_phase2_strict_bugfix(model, tokenizer, strict_loader, device, epochs=50)
    torch.save(model.state_dict(), "debug_model_phase2.pth")
    torch.save(model, "phase2mdl.pt")
    with open("phase2_tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    print("\ncomplex bugfixes\n")
    complex_examples = []
    for lang in languages:
        for buggy, fixed in load_plain_bugfix_pairs(dataset_path, lang):
            input_ids = tokenizer.encode(buggy, max_length=128, padding="max_length", truncation=True, return_tensors="pt")['input_ids'].squeeze(0)
            output_ids = tokenizer.encode(fixed, max_length=128, padding="max_length", truncation=True, return_tensors="pt")['input_ids'].squeeze(0)
            complex_examples.append({'input_ids': input_ids, 'labels': output_ids, 'language': lang, 'buggy_code': buggy, 'fixed_code': fixed})
        
        complex_examples.extend(create_pure_preservation_examples(tokenizer, lang, dataset_path, max_per_lang=20))
    complex_examples = ensure_tokenized_examples(complex_examples, tokenizer)
    complex_examples = filter_and_prepare_examples(complex_examples, tokenizer)
    complex_dataset = BugFixDataset(complex_examples, tokenizer)
    complex_loader = DataLoader(SafeDatasetWrapper(complex_dataset, tokenizer.vocab_size), batch_size=8, shuffle=True, collate_fn=custom_collate_fn)
    model = check_and_fix_embedding_size(model, tokenizer, device)
    model = train_phase3_complex_bugfix(model, tokenizer, complex_loader, device, epochs=50)
    torch.save(model.state_dict(), "final_phase3.pth")
    torch.save(model, "final_phase3_fullmodel.pt")
    with open("phase3_tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    print("Full model final_phase3_fullmodel.pt")



    
    print("main training complete.")
    return model





def phase_training(
    model, tokenizer, dataloader, device, epochs=20, reward_weight=1.0, 
    phase_name="PHASE", phase_goal="copy", template_penalty_weight=2.0, preservation_weight=2.0,
    val_dataloader=None, patience=40
):

    #methods for punishment and rewarding
    def extract_lines(code):
        return [l for l in code.splitlines() if l.strip()]

    def extract_ids(code):
        return set(re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\b', code))

    def is_template_like(text, input_text):
        templates = [
            "class Program", "static void Main", "namespace", "using System;",
            "def main():", "if __name__ == \"__main__\":", "package main", "func main() {"
        ]
        for t in templates:
            if t in text and t not in input_text:
                return True
        return False

    def identifier_preservation(input_text, output_text):
        input_ids = extract_ids(input_text)
        output_ids = extract_ids(output_text)
        if not input_ids:
            return 1.0
        return len(input_ids & output_ids) / len(input_ids)

    def hallucinated_identifier_penalty(input_text, output_text):
        input_ids = extract_ids(input_text)
        output_ids = extract_ids(output_text)
        hallucinated = len(output_ids - input_ids)
        if hallucinated > 0:
            return min(5.0, hallucinated * 2.0)
        return 0.0

    def hallucinated_output_penalty(input_text, pred_text):
        if pred_text.strip() in {"//", "", "/*", "#", ";"}:
            return 20.0
        alnum_ratio = sum(c.isalnum() for c in pred_text) / max(1, len(pred_text))
        if alnum_ratio < 0.2:
            return 10.0
        if len(pred_text.strip()) < max(3, len(input_text.strip()) // 2):
            return 10.0
        if pred_text and input_text.startswith(pred_text):
            return 10.0
        if pred_text.count('"') % 2 != 0 or pred_text.count("'") % 2 != 0:
            return 10.0
        if pred_text.count('(') != pred_text.count(')'):
            return 10.0
        if pred_text.count('{') != pred_text.count('}'):
            return 10.0
        return 0.0

    def end_of_block_penalty(pred_text, target_text):
        pred_lines = extract_lines(pred_text)
        target_lines = extract_lines(target_text)
        extra_lines = 0
        if len(pred_lines) > len(target_lines):
            for j in range(len(target_lines), len(pred_lines)):
                if pred_lines[j] not in target_lines:
                    extra_lines += 1
        return min(0.15, 0.03 * extra_lines)

    def incorrect_varname_penalty(input_text, target_text, pred_text):
        input_vars = extract_ids(input_text)
        target_vars = extract_ids(target_text)
        pred_vars = extract_ids(pred_text)
        incorrect_vars = pred_vars - (input_vars | target_vars)
        return min(0.15, 0.03 * len(incorrect_vars))

    def short_output_penalty(input_text, pred_text):
        in_len = len([l for l in input_text.splitlines() if l.strip()])
        out_len = len([l for l in pred_text.splitlines() if l.strip()])
        if out_len < max(2, int(1.0 * in_len)):
            return 5.0
        return 0.0

    def missing_line_penalty(input_text, pred_text):
        input_lines = set([l.strip() for l in input_text.splitlines() if l.strip()])
        pred_lines = set([l.strip() for l in pred_text.splitlines() if l.strip()])
        missing = input_lines - pred_lines
        if missing:
            return 5.0
        return 0.0

    def indentation_penalty(pred_text, target_text):
        pred_lines = [l for l in pred_text.splitlines() if l.strip()]
        target_lines = [l for l in target_text.splitlines() if l.strip()]
        if not pred_lines or not target_lines:
            return 0.0
        penalty = 0.0
        total = min(len(pred_lines), len(target_lines))
        for pl, tl in zip(pred_lines, target_lines):
            if (len(pl) - len(pl.lstrip())) != (len(tl) - len(tl.lstrip())):
                penalty += 1.0
        return penalty / total if total > 0 else 0.0

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.vocab["<pad>"])
    pad_token_id = tokenizer.vocab["<pad>"]
    eos_token_id = tokenizer.vocab["<eos>"]
    bos_token_id = tokenizer.vocab["<bos>"]
#update lr for phase 3, since it shoudl already have good knowledge
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    if phase_name == "PHASE 3":
        optimizer = torch.optim.AdamW(model.parameters(), lr=4e-5, weight_decay=0.01)
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_copy_reward = 0.0
        epoch_id_pres = 0.0
        epoch_template_penalty = 0.0
        epoch_hallucination_penalty = 0.0
        epoch_endblock_penalty = 0.0
        epoch_varname_penalty = 0.0
        epoch_missingline_penalty = 0.0
        epoch_short_penalty = 0.0
        epoch_indent_penalty = 0.0
        batch_count = 0
        all_preds = []
        all_targets = []
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            vocab_size = tokenizer.vocab_size
            input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
            labels = torch.clamp(labels, 0, vocab_size - 1)

            if not torch.all(labels[:, -1] == eos_token_id):
                labels[:, -1] = eos_token_id

            decoder_input = torch.full_like(labels, bos_token_id)
            decoder_input[:, 1:] = labels[:, :-1]
            target = labels

            try:
                logits = model(input_ids, decoder_input)
            except Exception as e:
                print(f"{e}")
                continue

            loss = loss_fn(logits.reshape(-1, logits.size(-1)), target.reshape(-1))

            pred_ids = logits.argmax(-1)
            batch_copy_reward = 0.0
            batch_id_pres = 0.0
            batch_template_penalty = 0.0
            batch_hallucination_penalty = 0.0
            batch_endblock_penalty = 0.0
            batch_varname_penalty = 0.0
            batch_missingline_penalty = 0.0
            batch_short_penalty = 0.0
            batch_force_copy = 0.0
            batch_similarity = 0.0
            batch_identical = 0.0
            batch_indent_penalty = 0.0
            batch_hallucinated_output_penalty = 0.0
            for i in range(pred_ids.size(0)):
                pred_trim = pred_ids[i]
                target_trim = target[i]
                special_ids = {pad_token_id, eos_token_id, bos_token_id}
                pred_list = [t for t in pred_trim.tolist() if t not in special_ids]
                target_list = [t for t in target_trim.tolist() if t not in special_ids]
                pred_text = tokenizer.decode(pred_list, skip_special_tokens=True).strip()
                target_text = tokenizer.decode(target_list, skip_special_tokens=True).strip()
                input_text = tokenizer.decode(input_ids[i], skip_special_tokens=True).strip()

                copy_reward = float(pred_list == target_list)
                id_pres = identifier_preservation(input_text, pred_text)
                if copy_reward == 1.0 or (id_pres > 0.95 and not is_template_like(pred_text, input_text)):
                    batch_copy_reward += 1.0
                else:
                    batch_copy_reward += 0.0

                batch_id_pres += id_pres

                template_penalty = float(is_template_like(pred_text, input_text))
                batch_template_penalty += template_penalty

                halluc_penalty = hallucinated_identifier_penalty(input_text, pred_text)
                batch_hallucination_penalty += halluc_penalty

                hallucinated_output = hallucinated_output_penalty(input_text, pred_text)
                batch_hallucinated_output_penalty += hallucinated_output

                endblock_penalty = end_of_block_penalty(pred_text, target_text)
                batch_endblock_penalty += endblock_penalty

                varname_penalty = incorrect_varname_penalty(input_text, target_text, pred_text)
                batch_varname_penalty += varname_penalty

                missingline_penalty = missing_line_penalty(input_text, pred_text)
                batch_missingline_penalty += missingline_penalty

                short_penalty = short_output_penalty(input_text, pred_text)
                batch_short_penalty += short_penalty

                if pred_text.strip() == input_text.strip():
                    batch_force_copy += 1.0

                similarity = SequenceMatcher(None, pred_text, target_text).ratio()
                batch_similarity += similarity

                if pred_text == target_text:
                    batch_identical += 1.0

                #indentation penalty penalty for all languages
                indent_penalty = indentation_penalty(pred_text, target_text)
                batch_indent_penalty += indent_penalty

                all_preds.append(int(pred_list == target_list))
                all_targets.append(1)

            batch_size = pred_ids.size(0)
            batch_copy_reward /= batch_size
            batch_id_pres /= batch_size
            batch_template_penalty /= batch_size
            batch_hallucination_penalty /= batch_size
            batch_endblock_penalty /= batch_size
            batch_varname_penalty /= batch_size
            batch_missingline_penalty /= batch_size
            batch_short_penalty /= batch_size
            batch_force_copy /= batch_size
            batch_similarity /= batch_size
            batch_identical /= batch_size
            batch_indent_penalty /= batch_size
            batch_hallucinated_output_penalty /= batch_size

            total_loss = loss \
                - reward_weight * (
                    3.0 * batch_copy_reward +
                    2.0 * batch_force_copy
                ) \
                + template_penalty_weight * batch_template_penalty \
                + 2.0 * batch_hallucination_penalty \
                + 5.0 * batch_hallucinated_output_penalty \
                + 1.5 * batch_endblock_penalty \
                + 1.5 * batch_varname_penalty \
                + preservation_weight * batch_missingline_penalty \
                + 3.0 * batch_short_penalty \
                + 4.0 * batch_indent_penalty  

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            epoch_copy_reward += batch_copy_reward
            epoch_id_pres += batch_id_pres
            epoch_template_penalty += batch_template_penalty
            epoch_hallucination_penalty += batch_hallucination_penalty
            epoch_endblock_penalty += batch_endblock_penalty
            epoch_varname_penalty += batch_varname_penalty
            epoch_missingline_penalty += batch_missingline_penalty
            epoch_short_penalty += batch_short_penalty
            epoch_indent_penalty += batch_indent_penalty
            batch_count += 1

        #validation
        val_loss = None
        if val_dataloader is not None:
            model.eval()
            total_val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for val_batch in val_dataloader:
                    pass
            val_loss = total_val_loss / max(1, val_batches)
            print(f"Validation Loss: {val_loss:.4f}")
#f1+confusion matrix 
        try:
            f1 = f1_score(all_targets, all_preds, zero_division=0)
            cm = confusion_matrix(all_targets, all_preds, labels=[1, 0])
            print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/batch_count:.4f} - Copy Reward: {epoch_copy_reward/batch_count:.4f} - ID Pres: {epoch_id_pres/batch_count:.4f} - Template Penalty: {epoch_template_penalty/batch_count:.4f} - Hallucination Penalty: {epoch_hallucination_penalty/batch_count:.4f} - Hallucinated Output Penalty: {batch_hallucinated_output_penalty:.4f} - EndBlock Penalty: {epoch_endblock_penalty/batch_count:.4f} - VarName Penalty: {epoch_varname_penalty/batch_count:.4f} - MissingLine Penalty: {epoch_missingline_penalty/batch_count:.4f} - Short Penalty: {epoch_short_penalty/batch_count:.4f} - Indent Penalty: {epoch_indent_penalty/batch_count:.4f} - F1: {f1:.4f}")
            print(f"Confusion Matrix (rows: true, cols: pred):\n{cm}")
        except Exception as e:
            print(f"Could not compute F1/confusion: {e}")

        #print sample for debugging
        model.eval()
        with torch.no_grad():
            batch = next(iter(dataloader))
            input_ids = batch['input_ids'][0].unsqueeze(0).to(device)
            labels = batch['labels'][0].unsqueeze(0).to(device)
            generated = torch.full((1, 1), bos_token_id, dtype=torch.long, device=device)
            pred_ids = []
            input_ids_flat = input_ids.squeeze(0)
            min_output_length = max(2, int(1.0 * input_ids_flat.size(0)))
            for step in range(input_ids_flat.size(0) * 2):
                try:
                    next_logits = model(input_ids, generated)[:, -1, :]
                except Exception as e:
                    break
                if pad_token_id in pred_ids:
                    break
                if len(pred_ids) >= 2 and pred_ids[-1] == pred_ids[-2]:
                    break
                token_counts = Counter(pred_ids)
                for tok, count in token_counts.items():
                    if count > 5:
                        break
                next_token = torch.argmax(next_logits).item()
                if next_token == eos_token_id and len(pred_ids) < min_output_length:
                    continue
                pred_ids.append(next_token)
                generated = torch.cat([generated, torch.tensor([[next_token]], device=device)], dim=1)
                if next_token == eos_token_id and len(pred_ids) >= min_output_length:
                    break
            if pred_ids and pred_ids[0] == bos_token_id:
                pred_ids = pred_ids[1:]
            print("Sample input     :", tokenizer.decode(input_ids_flat.tolist(), skip_special_tokens=True))
            print("Sample predicted :", tokenizer.decode(pred_ids, skip_special_tokens=True))
            print("Sample pred ids  :", pred_ids)
            print("Sample tokens    :", [tokenizer.id_to_token.get(idx, '?') for idx in pred_ids])
    print(f"\n{phase_name} training complete.\n")
    return model

#copy
def train_phase0_identity_copy(model, tokenizer, dataloader, device, epochs=120):
    return phase_training(
        model, tokenizer, dataloader, device, epochs=870,
        reward_weight=1.0, phase_name="PHASE 0", phase_goal="copy", template_penalty_weight=0.0
    )

#small bugs
def train_phase1_minimal_bugfix(model, tokenizer, dataloader, device, epochs=750):
    
    return phase_training(
        model, tokenizer, dataloader, device, epochs=500,
        reward_weight=0.8, phase_name="PHASE 1", phase_goal="minimal_bugfix", template_penalty_weight=1.0, preservation_weight=1.0
    )

#bigger bugs, ensures to block template like generation
def train_phase2_strict_bugfix(model, tokenizer, dataloader, device, epochs=500):

    return phase_training(
        model, tokenizer, dataloader, device, epochs=700,
        reward_weight=0.7, phase_name="PHASE 2", phase_goal="strict_bugfix", template_penalty_weight=2.0, preservation_weight=2.0
    )

#complex bugs
def train_phase3_complex_bugfix(model, tokenizer, dataloader, device, epochs=500):
    return phase_training(
        model, tokenizer, dataloader, device, epochs=750,
        reward_weight=0.6, phase_name="PHASE 3", phase_goal="complex_bugfix", template_penalty_weight=2.5, preservation_weight=3.0
    )



def custom_collate_fn(batch):
    batch_out = {}
    for key in batch[0]:
    #ensure tensors before stacking
        if isinstance(batch[0][key], torch.Tensor):
            batch_out[key] = torch.stack([item[key] for item in batch])
        else:
            batch_out[key] = [item[key] for item in batch]
    return batch_out

class DictDataset(torch.utils.data.Dataset):
    #simple dataset wrapper
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
    






class CodeMetricDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=256):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        code = self.examples[idx]['code']
        metrics = np.array([
            self.examples[idx]['complexity'],
            self.examples[idx]['readability'],
            self.examples[idx]['maintainability']
        ], dtype=np.float32)
        encoding = self.tokenizer.encode(
            code,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'metrics': torch.tensor(metrics)
        }





def identify_bugged_code(buggy_code, fixed_code, language, bug_type=None):
    augmented_examples = []
    
    #work out bug type
    if bug_type is None:
        if language.lower() == "python":
            if ":" not in buggy_code and ":" in fixed_code:
                bug_type = "syntax"
            elif "return" not in buggy_code and "return" in fixed_code:
                bug_type = "logic"
            elif "except" in fixed_code or "try" in fixed_code:
                bug_type = "exception"
            else:
                bug_type = "general"
        elif language.lower() == "go":
            if buggy_code.count("{") != buggy_code.count("}"):
                bug_type = "syntax"
            elif "nil" in fixed_code or "err" in fixed_code:
                bug_type = "error_handling"
            else:
                bug_type = "general"
        elif language.lower() in ["c#", "csharp"]:
            if ";" not in buggy_code and ";" in fixed_code:
                bug_type = "syntax"
            elif "null" in fixed_code or "exception" in fixed_code:
                bug_type = "null_handling"
            else:
                bug_type = "general"

    if language.lower() == "python":
        if bug_type == "syntax":
            #indentation errors
            lines = fixed_code.split("\n")
            for i in range(1, len(lines)):
                if lines[i].startswith("    "):
                    indented_code = fixed_code.replace(lines[i], lines[i][2:])  # Remove 2 spaces
                    augmented_examples.append({
                        "buggy_code": indented_code,
                        "fixed_code": fixed_code,
                        "bug_type": "syntax",
                        "language": language
                    })
                    break
            
            #missing colon
            if ":" in fixed_code:
                no_colon_code = fixed_code.replace(":", "", 1)
                augmented_examples.append({
                    "buggy_code": no_colon_code,
                    "fixed_code": fixed_code,
                    "bug_type": "syntax",
                    "language": language
                })
        
        elif bug_type == "logic":
            #off-by-one error
            if "range" in fixed_code:
                off_by_one_code = fixed_code.replace("range(", "range(1 + ", 1)
                augmented_examples.append({
                    "buggy_code": off_by_one_code,
                    "fixed_code": fixed_code,
                    "bug_type": "logic",
                    "language": language
                })
            
            #reversed comparison
            if "<" in fixed_code:
                reversed_code = fixed_code.replace("<", ">", 1)
                augmented_examples.append({
                    "buggy_code": reversed_code,
                    "fixed_code": fixed_code,
                    "bug_type": "logic",
                    "language": language
                })
            elif ">" in fixed_code:
                reversed_code = fixed_code.replace(">", "<", 1)
                augmented_examples.append({
                    "buggy_code": reversed_code,
                    "fixed_code": fixed_code,
                    "bug_type": "logic",
                    "language": language
                })
    
    elif language.lower() == "go":
        if bug_type == "error_handling":
            #missing error check
            if "if err != nil {" in fixed_code:
                no_err_check = fixed_code.replace("if err != nil {", "// Missing error check", 1)
                augmented_examples.append({
                    "buggy_code": no_err_check,
                    "fixed_code": fixed_code,
                    "bug_type": "error_handling",
                    "language": language
                })
            
            #not returning error
            if "return nil, err" in fixed_code:
                no_err_return = fixed_code.replace("return nil, err", "return nil, nil", 1)
                augmented_examples.append({
                    "buggy_code": no_err_return,
                    "fixed_code": fixed_code,
                    "bug_type": "error_handling",
                    "language": language
                })
        
        elif bug_type == "syntax":
            #missing brace
            if "{" in fixed_code and "}" in fixed_code:
                missing_brace = fixed_code.replace("}", "", 1)
                augmented_examples.append({
                    "buggy_code": missing_brace,
                    "fixed_code": fixed_code,
                    "bug_type": "syntax",
                    "language": language
                })
    
    elif language.lower() in ["c#", "csharp"]:
        if bug_type == "null_handling":
            #missing null check
            if "if (" in fixed_code and " == null)" in fixed_code:
                no_null_check = fixed_code.replace("if (", "// Missing null check\n//if (", 1)
                augmented_examples.append({
                    "buggy_code": no_null_check,
                    "fixed_code": fixed_code,
                    "bug_type": "null_handling",
                    "language": language
                })
        
        elif bug_type == "syntax":
            #missing semicolon
            if ";" in fixed_code:
                no_semicolon = fixed_code.replace(";", "", 1)
                augmented_examples.append({
                    "buggy_code": no_semicolon,
                    "fixed_code": fixed_code,
                    "bug_type": "syntax",
                    "language": language
                })
    
    return augmented_examples








def train_debug_model(classifier=None, tokenizer=None, id_to_language=None):
    print("\nStarting training!!\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = SimpleCodeTokenizer()
    code_model = SimpleCodeModel(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.vocab["<pad>"],
        bos_token_id=tokenizer.vocab["<bos>"],
        eos_token_id=tokenizer.vocab["<eos>"],
        d_model=1024,
        nhead=16,
        num_encoder_layers=8,
        num_decoder_layers=8,
        dim_feedforward=4096,
        dropout=0.25,
    ).to(device)
    #delete existing models
    if os.path.exists(FINAL_DEBUG_MODEL_PATH):
        os.remove(FINAL_DEBUG_MODEL_PATH)
       

    print(f"device: {device}")
    if hasattr(code_model, '_expand_embeddings'):
        if code_model.vocab_size != tokenizer.vocab_size:
            code_model._expand_embeddings(tokenizer.vocab_size, device)
#get bugfix examples
    debug_examples = extract_debug_examples(DEBUG_DATASET_PATH)
    bugfix_examples = [ex for ex in debug_examples 
                      if "buggy_code" in ex and "fixed_code" in ex 
                      and ex["buggy_code"].strip() != ex["fixed_code"].strip()]
    languages = sorted(list(set(ex["language"].lower() for ex in bugfix_examples)))

    # calculate weigths to balance learning
    language_counts = Counter([ex["language"].lower() for ex in bugfix_examples])
    max_count = max(language_counts.values()) if language_counts else 1
    language_weights = {lang: max_count / (count + 1) for lang, count in language_counts.items()}


    code_tokenizer = SimpleCodeTokenizer()

     #create weighted examples
    weighted_examples = []
    for ex in bugfix_examples:
        lang = ex["language"].lower()
        repeats = max(1, int(language_weights.get(lang, 1) + 0.5))
        augmented = identify_bugged_code(ex["buggy_code"], ex["fixed_code"], lang)
        
        if augmented:
            weighted_examples.extend(augmented)
            weighted_examples.append(ex)
        else:
            weighted_examples.extend([ex] * repeats)
    
    #create dataset with the new examples
    weighted_examples = ensure_tokenized_examples(weighted_examples, code_tokenizer)
    bugfix_dataset = BugFixDataset(weighted_examples, code_tokenizer)
    bugfix_train_indices, bugfix_val_indices = train_test_split(
        range(len(bugfix_dataset)), test_size=0.15, random_state=42)
    
    #language specific validation
    language_val_indices = {}
    for lang in languages:
        lang_indices = [i for i, idx in enumerate(bugfix_val_indices) 
                      if weighted_examples[idx]["language"].lower() == lang]
        language_val_indices[lang] = [bugfix_val_indices[i] for i in lang_indices]
        print(f"Created validation set for {lang} with {len(language_val_indices[lang])} examples")
    
    #dataloaders
    bugfix_train_dataloader = DataLoader(
        torch.utils.data.Subset(bugfix_dataset, bugfix_train_indices),
        batch_size=4, shuffle=True, num_workers=2
    )
    bugfix_val_dataloader = DataLoader(
        torch.utils.data.Subset(bugfix_dataset, bugfix_val_indices),
        batch_size=4, shuffle=False, num_workers=1
    )

    #recreate model
    code_model = SimpleCodeModel(
        vocab_size=code_tokenizer.vocab_size,
        pad_token_id=code_tokenizer.vocab["<pad>"],
        bos_token_id=code_tokenizer.vocab["<bos>"],
        eos_token_id=code_tokenizer.vocab["<eos>"],
        d_model=1024,
        nhead=16,
        num_encoder_layers=8,
        num_decoder_layers=8,
        dim_feedforward=4096,
        dropout=0.25,
    ).to(device)

    print("\nschedule training started!\n")
    code_model = training_scheduler(
    code_model,  
    code_tokenizer,  
    bugfix_train_dataloader,  
    bugfix_val_dataloader,    
    device,
    epochs=100,  
    patience=30,
    learning_rate=1e-5  
    )
    
    
    print("\nadding metrics to model\n")
    
    try:
        code_model = torch.load(r"C:\Users\James\Downloads\dev3\final_phase3_fullmodel.pt", map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"model loading error")
    
    
    code_model, metrics_head = train_metrics(
        code_model, 
        code_tokenizer, 
        bugfix_dataset, 
        weighted_examples, 
        languages, 
        device,
        accuracy_target=0.80  
    )
    torch.save(metrics_head.state_dict(), "quality_metrics_head.pth") 
    torch.save(code_model, "modelmetrics.pt")
    
    print("\nmetrics test\n")
    try:
        for language in ["python", "go", "c#"]:
            # Create simple demo code for each language
            if language == "python":
                demo_code = """
def calculate_sum(numbers):
    total = 0
    for num in numbers:
        total += num
    return total
                """
            elif language == "go":
                demo_code = """
package main

import "fmt"

func calculateSum(numbers []int) int {
    total := 0
    for _, num := range numbers {
        total += num
    }
    return total
}
                """
            else:  # C#
                demo_code = """
using System;
using System.Linq;

public class Calculator {
    public static int CalculateSum(int[] numbers) {
        int total = 0;
        foreach (int num in numbers) {
            total += num;
        }
        return total;
    }
}
                """

            #score the code
            results = SimpleCodeModel.generate_with_metrics(
                code_model, metrics_head, code_tokenizer, 
                demo_code, language, device
            )

            print(f"\n{language} Code Quality Assessment:")
            def print_metric_block(title, key):
                print(title)
                metrics = results.get(key, results.get("metrics", {}))
                for cat in ["security", "efficiency", "readability", "structure", "general"]:
                    val = metrics.get(cat, "N/A")
                    if isinstance(val, (float, int)):
                        print(f"  {cat.capitalize()}: {val:.2f}")
                    else:
                        print(f"  {cat.capitalize()}: {val}")

            print_metric_block("Original Code Metrics:", "original_metrics")
            print_metric_block("Improved Code Metrics:", "improved_metrics")

            

    except Exception as e:
        print(f"error in code quality test")
      

    # Save full model
    torch.save({
        'model': code_model,
        'classifier_head': classifier,  
        'tokenizer': code_tokenizer,
        'languages': languages,
        'language_to_id': {lang: i for i, lang in enumerate(languages)},
        'id_to_language': id_to_language if id_to_language is not None else {i: lang for i, lang in enumerate(languages)}
    }, "final_model_complete.pth") 
   
    torch.save(code_tokenizer, "final_tokenizer.pth")
    with open("metrics_tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    print(f"\nTraining complete! :)")

    return code_model, code_tokenizer, languages

def score_code_quality(code, language, device=None):
    language = language.lower()
    if language == "c#":
        language = "csharp"

    #default scores
    scores = {
        "security": 50,
        "maintainability": 50,
        "performance": 50,
        "readability": 50,
        "functionality": 50,
        "explanation": {}
    }

    try:
        if language == "python":
         
            security_score = 90
            security_issues = []
            if any(x in code for x in ['eval(', 'exec(', 'os.system', 'subprocess']):
                security_score -= 40
                security_issues.append("Use of dangerous function")
            if 'password' in code or 'api_key' in code:
                security_score -= 20
                security_issues.append("Hardcoded credentials")
            scores["security"] = max(0, min(100, security_score))

            try:
                mi = mi_visit(code, True)
                maintainability = mi['mi']
                scores["maintainability"] = max(10, min(100, maintainability + random.randint(-5, 5)))
            except Exception:
                scores["maintainability"] = 50

         
            try:
                blocks = cc_visit(code)
                complexity = sum(b.complexity for b in blocks) / (len(blocks) or 1)
                perf_score = max(10, 100 - complexity * 10 + random.randint(-5, 5))
                scores["performance"] = min(100, perf_score)
            except Exception:
                scores["performance"] = 50

       
            lines = [line for line in code.split('\n') if line.strip()]
            comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
            comment_ratio = comment_lines / max(1, len(lines))
            readability_score = 60
            if 0.1 <= comment_ratio <= 0.3:
                readability_score += 15
            elif comment_ratio < 0.05:
                readability_score -= 10
            long_lines = sum(1 for line in lines if len(line.strip()) > 79)
            if long_lines > 0:
                readability_score -= min(long_lines * 2, 15)
            scores["readability"] = max(10, min(100, readability_score + random.randint(-5, 5)))

           
            functionality_score = 60
            if 'def ' in code and 'return' in code:
                functionality_score += 20
            if 'TODO' in code or 'FIXME' in code:
                functionality_score -= 20
            scores["functionality"] = max(10, min(100, functionality_score + random.randint(-5, 5)))
            scores["explanation"] = {
                "security": {"issues": security_issues},
                "maintainability": {"mi": scores["maintainability"]},
                "performance": {"complexity": scores["performance"]},
                "readability": {"comment_ratio": comment_ratio},
                "functionality": {"heuristic": functionality_score}
            }
        else:        
            code_len = len(code)
            scores["security"] = 90 if 'eval' not in code and 'exec' not in code else 40
            scores["maintainability"] = 100 - min(80, code.count(';') // 5)
            scores["performance"] = 100 - min(90, code_len // 10)
            scores["readability"] = 100 - min(80, code.count('\n') // 2)
            scores["functionality"] = 90 if 'return' in code else 60
            #add small bit of noise
            for k in scores:
                if k != "explanation":
                    scores[k] = max(10, min(100, scores[k] + random.randint(-5, 5)))
            scores["explanation"] = {}
    except Exception as e:
        print(f"{str(e)}")
    return scores


def main():
    
    print("starting training")
    

    classifier = None
    tokenizer = None
    id_to_language = None
    
    print("debugging training")  
    train_debug_model(classifier, tokenizer, id_to_language)
    print("Training complete")

if __name__ == "__main__":
    main()
    print("finished")