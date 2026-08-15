import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import os

base_model_name = "Qwen/Qwen3-4B"
adapter_repo_id = "AxelDlv00/ToxiFrench" 
target_adapter = "SOAP-DWL-DPO" 

tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokens = ["<think>", "</think>"]
tokenizer.add_special_tokens({"additional_special_tokens": tokens})

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    trust_remote_code=True,
    device_map="auto"
)

tokenizer_vocab_size = len(tokenizer)
model_embedding_size = model.get_input_embeddings().weight.size(0)

if model_embedding_size != tokenizer_vocab_size:
    print(f"Syncing vocab: {model_embedding_size} -> {tokenizer_vocab_size}")
    model.resize_token_embeddings(tokenizer_vocab_size)

model = PeftModel.from_pretrained(model, adapter_repo_id, subfolder=target_adapter)
model.eval()

text = "Je ne supporte plus ton comportement, tu es vraiment un idiot !"
prompt = f"Message:\n{text}\n\nAnalyse:\n"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs, 
        max_new_tokens=512, 
        temperature=0.7, 
        do_sample=True,
        repetition_penalty=1.1
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=False))