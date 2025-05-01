from pathlib import Path

def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 10,
        "d_model": 512,
        "lr": 10**-4,
        "seq_len": 350,
        "model_folder": "models",
        "model_basename": "transformer_codedetect",
        "preload": None,
        "tokenizer_file": "tokenizer.json",
        "data_files": "rosetta_code_dataset.json",  
        "split": "train", 
        "experiment_name": "transformer_codedetect",
    }

def get_weights_file_path(config, epoch_str):  
    modelFolder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f"{model_basename}_{epoch_str}.pt"  
    return str(Path('.') / modelFolder / model_filename)
