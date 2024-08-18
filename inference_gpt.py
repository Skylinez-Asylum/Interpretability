import torch
import tiktoken
from model import GPT, GPTConfig

def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_config = checkpoint['config']
    model = GPT(model_config)
    model.load_state_dict(checkpoint['model'])
    return model

def generate(model, prompt, max_new_tokens, temperature=1.0, top_k=None):
    model.eval()
    enc = tiktoken.get_encoding("gpt2")
    input_ids = torch.tensor(enc.encode(prompt)).unsqueeze(0)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = input_ids if input_ids.size(1) <= model.config.block_size else input_ids[:, -model.config.block_size:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = torch.nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat((input_ids, idx_next), dim=1)

    generated_text = enc.decode(input_ids[0].tolist())
    return generated_text

if __name__ == "__main__":
    checkpoint_path = "log/model_19072.pt" 
    model = load_model(checkpoint_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    prompt = "Once upon a time"
    max_new_tokens = 100
    temperature = 0.8
    top_k = 40

    generated_text = generate(model, prompt, max_new_tokens, temperature, top_k)
    print(generated_text)