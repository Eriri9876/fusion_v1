from pathlib import Path
import re

p = Path("layers/MLF_backbone.py")
lines = p.read_text(encoding="utf-8").splitlines(True)

def remove_lines_containing(keys):
    global lines
    out = []
    for l in lines:
        if any(k in l for k in keys):
            continue
        out.append(l)
    lines = out

# 先粗暴清掉我们之前插入的标记行，避免越插越乱
remove_lines_containing([
    "FORCE_TOP3_NEIGHBORS", "FORCE_TOPK_NEIGHBORS_V2",
    "DEBUG_BACKBONE_MASK_ONCE", "DEBUG_BACKBONE_MASK_ONCE2",
    "[mask@backbone]", "[mask@backbone2]"
])

einsum_pat = "torch.einsum('bnpd, mn -> bmpd'"
idx = None
for i,l in enumerate(lines):
    if einsum_pat in l:
        idx = i
        break
if idx is None:
    raise SystemExit("cannot find einsum line")

indent = re.match(r'^(\s*)', lines[idx]).group(1)

patch = [
    indent + "# ---- FORCE_TOPK_NEIGHBORS_V2 ----\n",
    indent + "cm = channel_mask.clone()\n",
    indent + "cm.fill_diagonal_(0)\n",
    indent + "k_keep = min(3, cm.shape[0]-1)\n",
    indent + "if k_keep > 0:\n",
    indent + "    vals, inds = torch.topk(cm, k=k_keep, dim=1)\n",
    indent + "    sparse = torch.zeros_like(cm)\n",
    indent + "    sparse.scatter_(1, inds, vals)\n",
    indent + "    cm = sparse / (sparse.sum(dim=1, keepdim=True) + 1e-6)\n",
    indent + "channel_mask = cm\n",
    indent + "if not hasattr(self, 'DEBUG_BACKBONE_MASK_ONCE2'):\n",
    indent + "    self.DEBUG_BACKBONE_MASK_ONCE2 = True\n",
    indent + "    with torch.no_grad():\n",
    indent + "        nz = (channel_mask.abs() > 1e-12).float().mean().item()\n",
    indent + "        row_nz = (channel_mask.abs() > 1e-12).sum(dim=1).unique().detach().cpu().tolist()\n",
    indent + "        print(f'[mask@backbone2] min={channel_mask.min().item():.4f} max={channel_mask.max().item():.4f} nonzero_ratio={nz:.3f} row_nz_unique={row_nz}')\n",
]

lines = lines[:idx] + patch + lines[idx:]
p.write_text("".join(lines), encoding="utf-8")
print("patched backbone: FORCE_TOPK_NEIGHBORS_V2")
