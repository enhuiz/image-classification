import sys
import torch

assert len(sys.argv) == 2, "Give one path"

state_dict = torch.load(sys.argv[1], "cpu")["module"]

τs = []
for key, val in state_dict.items():
    if "log_one_div_by_τ" in key:
        τ = (1 / val.exp()).item()
        τs.append(τ)
        print(key, val.item(), "τ", τ)
    elif "τ" in key:
        τ = val.item()
        τs.append(τ)
        print(key, τ)

τs = torch.tensor(τs)
print(f"{τs.mean().item():.3g}±{τs.std().item():.3g}")
