import torch
import torch.nn.functional as F

from config import TEMPERATURE, MAX_NEW


@torch.no_grad()
def generate(model, enc, prompt, max_new=MAX_NEW,
             temperature=TEMPERATURE, device="cuda"):
    model.eval()
    ids = enc.encode_ordinary(prompt)
    ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(max_new):
        logits  = model(ids)
        logits  = logits[0, -1, :] / temperature
        probs   = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        ids     = torch.cat([ids, next_id.unsqueeze(0)], dim=1)

    return enc.decode(ids[0].tolist())
