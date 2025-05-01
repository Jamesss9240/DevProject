import torch
import torch.nn as nn
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import math
import sys
import traceback
import re
import pickle
from collections import Counter
from difflib import SequenceMatcher

sys.path.append(r"C:\Users\James\Downloads\dev3\code-improvement-web-app")
from Code_improvement_transformer import model_Code
sys.modules['model_Code'] = model_Code
from Code_improvement_transformer.model_Code import SimpleCodeModel, SimpleCodeTokenizer, LanguageClassifierHead
from Code_improvement_transformer.train4debug import QualityMetricsHead

app = Flask(__name__)
CORS(app)

#config to pick model(mostly for debugging, for ugeneral full usage select any interger that isnt 0,1,2 or 3, since this loads the full metrics model)
PHASE = 7  
if PHASE == 0:
    model_path = r"C:\Users\James\Downloads\dev3\full0_alllangs.pt"
    tokenizer_path = r"C:\Users\James\Downloads\dev3\phase0_tokenizer.pkl"
elif PHASE == 1:
    model_path = r"C:\Users\James\Downloads\dev3\phase1mdl.pt"
    tokenizer_path = r"C:\Users\James\Downloads\dev3\phase1_tokenizer.pkl"
elif PHASE == 2:
    model_path = r"C:\Users\James\Downloads\dev3\phase2mdl.pt"
    tokenizer_path = r"C:\Users\James\Downloads\dev3\phase2_tokenizer.pkl"
elif PHASE == 3:
    model_path = r"C:\Users\James\Downloads\dev3\final_phase3_fullmodel.pt"
    tokenizer_path = r"C:\Users\James\Downloads\dev3\phase3_tokenizer.pkl"
else:
    model_path = r"C:\Users\James\Downloads\dev3\modelmetrics.pt"
    tokenizer_path = r"C:\Users\James\Downloads\dev3\metrics_tokenizer.pkl"

languages = ['python', 'go', 'csharp', 'c#']
language_to_id = {lang: idx for idx, lang in enumerate(languages)}
id_to_language = {idx: lang for idx, lang in enumerate(languages)}

model = None
language_classifier = None

def initialize_tokenizer(tokenizer_path):
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    return tokenizer

tokenizer = initialize_tokenizer(tokenizer_path)



def extract_fixed_code(output_text, original_code):
    #extra prompting to ensure correct language generation
    prefixes = [
        "### PYTHON FIX BUGS ONLY ###",
        "### GO FIX BUGS ONLY ###",
        "### C# FIX BUGS ONLY ###",
        "### CODE FIX BUGS ONLY ###",
        "# FIXED CODE:",
        "## FIXED CODE:",
        "// FIXED CODE:",
        "FIXED CODE:"
    ]
    for prefix in prefixes:
        if prefix in output_text:
            output_text = output_text.split(prefix, 1)[-1].strip()
 
    if len(output_text) > len(original_code) * 2 or len(output_text) < len(original_code) * 0.5:

        segments = []
        lines = output_text.split("\n")
        for i in range(len(lines)):
            segment = "\n".join(lines[i:i+len(original_code.split("\n"))])
            sim = SequenceMatcher(None, segment, original_code).ratio()
            segments.append((segment, sim))
        segments.sort(key=lambda x: x[1], reverse=True)
        if segments and segments[0][1] > 0.3:
            return segments[0][0]
    return output_text

def improved_generate_with_language_priming(model, tokenizer, buggy_code, language, device, max_length=128):


    # Detect if this is a metrics model (not phase 0/1/2/3)
    metrics_model = not (PHASE in [0, 1, 2, 3])

    prompt = buggy_code
    encoded = tokenizer.encode(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    input_ids = torch.clamp(input_ids, 0, tokenizer.vocab_size - 1)
    input_ids_flat = input_ids[0]

    bos_token_id = tokenizer.vocab.get("<bos>", 0)
    eos_token_id = tokenizer.vocab.get("<eos>", 1)
    pad_token_id = tokenizer.vocab.get("<pad>", 2)

    model.eval()
    with torch.no_grad():
        generated = torch.full((1, 1), bos_token_id, dtype=torch.long, device=device)
        pred_ids = []
        for step in range(max_length):
            try:
                logits = model(input_ids, generated)
                next_logits = logits[0, -1, :].clone()
            except Exception as e:
                print(f"generate failed: {e}")
                break

            # small bit of pen logic for direction
            if pad_token_id in pred_ids:
                next_logits[pad_token_id] -= 100.0
            if len(pred_ids) >= 2 and pred_ids[-1] == pred_ids[-2]:
                next_logits[pred_ids[-1]] -= 20.0
            token_counts = Counter(pred_ids)
            for tok, count in token_counts.items():
                if count > 3:
                    next_logits[tok] -= (count - 3) * 5.0
            if step < input_ids_flat.size(0):
                correct_token = input_ids_flat[step].item()
                for idx in range(next_logits.size(0)):
                    if idx != correct_token:
                        next_logits[idx] -= 5.0

            next_token = torch.argmax(next_logits).item()
            pred_ids.append(next_token)
            generated = torch.cat([generated, torch.tensor([[next_token]], device=device)], dim=1)
            if next_token == eos_token_id:
                break

        #remove bos token if first
        if pred_ids and pred_ids[0] == bos_token_id:
            pred_ids = pred_ids[1:]

        raw_output = tokenizer.decode(pred_ids, skip_special_tokens=True)
        fixed_code = extract_fixed_code(raw_output, buggy_code)
        if not fixed_code.strip() or fixed_code.strip() == "...":
            fixed_code = buggy_code

    # generate metrics
    metrics_result = None
    if metrics_model and hasattr(model, "generate_with_metrics"):
        try:
            metrics_head = getattr(model, "metric_head", None)
            if metrics_head is not None:
                metrics_result = model.generate_with_metrics(
                    metrics_head, tokenizer, buggy_code, language, device
                )
            else:
                
                metrics_result = {"metrics": {}}
        except Exception as e:
            print(f"metrics err: {e}")

    # return fixed code+metrics if they exist
    result = {
        "fixed_code": fixed_code
    }
    if metrics_result is not None:
        result["quality_metrics"] = metrics_result.get("metrics") or metrics_result.get("improved_metrics") or {}
        result["metrics_full_result"] = metrics_result

    return result

def detect_language(code):
    #language detection, since model has has issues, makes results more stable for now
    if "def " in code and ":" in code:
        return "python"
    if "func " in code and "{" in code:
        return "go"
    if "using System" in code or "Console.WriteLine" in code:
        return "csharp"
    if "public class" in code or "namespace" in code:
        return "csharp"
    return "python"




def load_model():
    global model, language_classifier, tokenizer

    if os.path.exists(model_path):
        print(f"Loading full model from {model_path}")
        try:
            tokenizer = initialize_tokenizer(tokenizer_path)
            from torch.serialization import add_safe_globals
            from Code_improvement_transformer.model_Code import SimpleCodeModel
            add_safe_globals([SimpleCodeModel])
            model = torch.load(model_path, map_location='cpu', weights_only=False)
            print("model loaded")

            language_classifier = getattr(model, 'language_classifier', None)


            if PHASE not in [0, 1, 2, 3]:
                from Code_improvement_transformer.train4debug import QualityMetricsHead
                metrics_head_path = r"C:\Users\James\Downloads\dev3\quality_metrics_head.pth"
                if os.path.exists(metrics_head_path):
                    #load metrics
                    metrics_head = QualityMetricsHead(input_dim=model.d_model, hidden_dim=1024)
                    metrics_head.load_state_dict(torch.load(metrics_head_path, map_location='cpu'))
                    metrics_head.eval()
                    model.metric_head = metrics_head
                    print("metrics added and loaded")
                else:
                    print("metrics not found")
        

            if model is None:
                print("model failed")
                return False

            try:
                param_count = sum(p.numel() for p in model.parameters())
                non_zero_params = sum((p != 0).sum().item() for p in model.parameters())

                model.eval()
                if language_classifier is not None:
                    language_classifier.eval()
            except Exception as e:
                print(f"Error: {e}")
                model = None
                return False

            
            return True

        except Exception as e:
            print(f"err loading mdl: {str(e)}")
            traceback.print_exc()
            model = None
            return False
    else:
        print(f"mdl not found {model_path}")
        model = None
        return False

def test_model():
    if model is not None:
        test_codes = {
            'python': "def hello():\n    print('Hello world')",
            'go': "package main\n\nfunc main() {\n    fmt.Println(\"Hello World\")\n}",
            'csharp': "using System;\nclass Program {\n    static void Main() {\n        Console.WriteLine(\"Hello World\");\n    }\n}"
        }
        try:
            for true_lang, test_code in test_codes.items():
                print(f"\nTesting language detection for {true_lang}...")
                heuristic_lang = detect_language(test_code)
                print(f"Heuristic detection says: {heuristic_lang}")

            buggy_code_tests = {
                'python': [
                    "def hello():\n    print('Hello world'",  
                    "for i in range(10)\n    print(i)",  
                    "if x > 5\n    return True"  
                ],
                'go': [
                    "func main() {\n    fmt.Println(\"Hello\"",
                    "package main\n\nfunc add(a, b int) {\n    return a + b"  
                ],
                'csharp': [
                    "Console.WriteLine(\"Hello\"",  
                    "public class Test {\n    public void Method() {\n        var x = 5\n    }" 
                ]
            }

            print("\n testing")
            for lang, bugs in buggy_code_tests.items():
                for buggy in bugs[:1]:
                    print(f"\nFixing {lang} code: {buggy}")
                    fixed = improved_generate_with_language_priming(model, tokenizer, buggy, lang, next(model.parameters()).device)
                    print(f"FIXED: {fixed}")

            print("\ntesting complete")
            return True
        except Exception as e:
            print(f"error: {str(e)}")
            traceback.print_exc()
            return False
    else:
        print("model failed load")
        return False

load_successful = load_model()
if load_successful:
    test_result = test_model()


@app.route('/process_code', methods=['POST'])
def process_code():
    try:
        data = request.get_json()
        code = data['code']

        if model is None:
            return jsonify({'output': "Model not loaded. Check server logs for details."})

        device = next(model.parameters()).device
        language = detect_language(code)
        fixed_code = improved_generate_with_language_priming(model, tokenizer, code, language, device)
        return jsonify({'output': fixed_code})
    except:
        return jsonify({'error'})

@app.route('/debug_code', methods=['POST'])
def debug_code():
    try:
        data = request.get_json()
        code = data['code']
        language = data.get('language', None)

        if model is None:
            return jsonify({
                'success': False,
                'original': code,
                'fixed': None,
                'language': None,
                'error': "Model not loaded. Check server logs for details."
            })

        device = next(model.parameters()).device
        if language is None:
            language = detect_language(code)
            print(f"Detected language: {language}")

        print(f"language used for fixing:: {language}")
        result = improved_generate_with_language_priming(model, tokenizer, code, language, device)
        response = {
            'success': True,
            'original': code,
            'fixed': result.get('fixed_code', ''),
            'language': language,
            'quality_metrics': result.get('quality_metrics', {})
        }
        return jsonify(response)
    except Exception as e:
        print(f"err in /debug code: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'original': code if 'code' in locals() else None,
            'error': str(e)
        })

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'status': 'online',
        'message': 'api is running!'
    })

if __name__ == '__main__':
    app.run(debug=True)