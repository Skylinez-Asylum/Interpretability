import os
import torch
import tiktoken
import numpy as np
from model import GPT, GPTConfig
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_config = checkpoint['config']
    model = GPT(model_config)
    model.load_state_dict(checkpoint['model'])
    return model

def get_mlp_activation(model, x):
    activations = []
    def hook_fn(module, input, output):
        activations.append(output.detach())
    
    hooks = []
    for block in model.transformer.h:
        hook = block.mlp.register_forward_hook(hook_fn)
        hooks.append(hook)
    
    model(x)
    
    for hook in hooks:
        hook.remove()
    
    return activations

def collect_activations(model, dataloader, save_dir, batch_size=32, max_samples=None):
    model.eval()
    enc = tiktoken.get_encoding("gpt2")
    device = next(model.parameters()).device

    sample_count = 0
    for batch in tqdm(dataloader):
        texts = batch['text']
        for text in texts:
            input_ids = torch.tensor(enc.encode(text)[:model.config.block_size]).unsqueeze(0).to(device)
            
            with torch.no_grad():
                activations = get_mlp_activation(model, input_ids)
            
            tokens = enc.decode_batch(input_ids.cpu().tolist()[0])
            
            save_path = os.path.join(save_dir, f"activations_{sample_count}.txt")
            with open(save_path, 'w') as f:
                for idx, (token, acts) in enumerate(zip(tokens, zip(*activations))):
                    f.write(f"Token {idx}: {token}\n")
                    for layer, act in enumerate(acts):
                        act_mean = act.mean().item()
                        act_std = act.std().item()
                        f.write(f"  Layer {layer} - Mean: {act_mean:.4f}, Std: {act_std:.4f}\n")
                    f.write("\n")
            
            sample_count += 1
            if max_samples and sample_count >= max_samples:
                return
        
        if max_samples and sample_count >= max_samples:
            break

if __name__ == "__main__":
    checkpoint_path = "log/model_19072.pt"  
    model = load_model(checkpoint_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    dataset_name = "wikitext"  
    dataset = load_dataset(dataset_name, "wikitext-103-raw-v1", split="train")
    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    save_dir = "activations"
    os.makedirs(save_dir, exist_ok=True)
    max_samples = 1000 
    collect_activations(model, dataloader, save_dir, batch_size, max_samples)

    print(f"Activations saved to {save_dir}")