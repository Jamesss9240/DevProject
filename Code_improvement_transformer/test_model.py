import pytest
import torch
from model_Code import SimpleCodeModel, SimpleCodeTokenizer
from train4debug import phase_training

@pytest.fixture
def my_tokenizer():
    return SimpleCodeTokenizer()

@pytest.fixture
def my_model(my_tokenizer):
    return SimpleCodeModel(
        vocab_size=my_tokenizer.vocab_size,
        pad_token_id=my_tokenizer.vocab["<pad>"],
        bos_token_id=my_tokenizer.vocab["<bos>"],
        eos_token_id=my_tokenizer.vocab["<eos>"],
        d_model=32,
        nhead=2,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=64,
        dropout=0.1,
    )

@pytest.fixture
def my_dummy_dataloader(my_tokenizer):
    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self): return 4
        def __getitem__(self, idx):
            code = "def test():\n    return 42"
            encoded = my_tokenizer.encode(code, max_length=32, return_tensors=None)
            return {
                "input_ids": torch.tensor(encoded["input_ids"][0], dtype=torch.long),
                "labels": torch.tensor(encoded["input_ids"][0], dtype=torch.long),
                "language": "python"
            }
    return torch.utils.data.DataLoader(DummyDataset(), batch_size=2)

def test_tokenizer_encode_decode(my_tokenizer):
    text = "def test():\n    return 42"
    encoded = my_tokenizer.encode(text, max_length=32)
    decoded = my_tokenizer.decode(encoded["input_ids"][0])
    assert "test" in decoded

def test_model_forward_shape(my_model, my_tokenizer):
    code = "def test():\n    return 42"
    encoded = my_tokenizer.encode(code, max_length=32)
    input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long)
    decoder_input = input_ids.clone()
    logits = my_model(input_ids, decoder_input)
    assert logits.shape[0] == input_ids.shape[0]
    assert logits.shape[-1] == my_model.vocab_size

def test_model_generate(my_model, my_tokenizer):
    code = "def test():\n    return 42"
    encoded = my_tokenizer.encode(code, max_length=32)
    input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long)
    if hasattr(my_model, "generate"):
        output = my_model.generate(input_ids)
        assert output.shape[0] == input_ids.shape[0]
        assert (output != my_model.pad_token_id).any()

def test_phase_training_runs(my_model, my_tokenizer, my_dummy_dataloader):
    device = torch.device("cpu")
    trained_model = phase_training(
        my_model, my_tokenizer, my_dummy_dataloader, device, epochs=1, phase_name="PHASE 0"
    )
    assert isinstance(trained_model, SimpleCodeModel)

def test_phase_training_perfect_copy_reward(my_model, my_tokenizer, my_dummy_dataloader):
    device = torch.device("cpu")
    phase_training(
        my_model, my_tokenizer, my_dummy_dataloader, device, epochs=1, phase_name="PHASE 0"
    )

def test_phase_training_perfect_fix_reward(my_model, my_tokenizer, my_dummy_dataloader):
    device = torch.device("cpu")
    phase_training(
        my_model, my_tokenizer, my_dummy_dataloader, device, epochs=1, phase_name="PHASE 1"
    )



def test_tokenizer_special_tokens(my_tokenizer):
    for tok in ["<pad>", "<unk>", "<eos>", "<bos>"]:
        assert tok in my_tokenizer.vocab

def test_model_decode_handles_special_tokens(my_tokenizer):
    if "<byte_100>" in my_tokenizer.vocab:
        ids = [my_tokenizer.vocab["<bos>"], my_tokenizer.vocab["<byte_100>"], my_tokenizer.vocab["<eos>"]]
        decoded = my_tokenizer.decode(ids)
        assert isinstance(decoded, str)
    else:
        pytest.skip("Tokenizer missingf token")