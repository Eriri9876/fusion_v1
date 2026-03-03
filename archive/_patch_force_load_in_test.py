from pathlib import Path
import re

p = Path(r"exp/exp_main_public.py")
s = p.read_text(encoding="utf-8", errors="ignore").splitlines(True)

marker = "# [PATCH] FORCE_LOAD_CKPT_IN_TEST"
if any(marker in ln for ln in s):
    print("SKIP: FORCE_LOAD_CKPT_IN_TEST already applied.")
    raise SystemExit(0)

# 找到 def test(self, setting, test=0): 或 def test(self, setting, ...):
def_idx = None
for i, ln in enumerate(s):
    if re.match(r"^\s*def\s+test\s*\(", ln):
        def_idx = i
        break
if def_idx is None:
    raise SystemExit("ERROR: cannot find def test(...) in exp_main_public.py")

# 找到 test 函数体第一行（下一行的缩进）
body_idx = def_idx + 1
while body_idx < len(s) and s[body_idx].strip() == "":
    body_idx += 1
indent = re.match(r"^(\s*)", s[body_idx]).group(1)

# 插入强制加载代码（依赖 torch / os，一般文件里已有 import torch；没有就补）
need_os = not any(re.match(r"^\s*import\s+os\s*$", x) for x in s[:120])
insert = []
insert.append(indent + marker + "\n")
if need_os:
    insert.append(indent + "import os\n")
insert += [
    indent + "try:\n",
    indent + "    # build ckpt dir = checkpoints/setting\n",
    indent + "    ckpt_dir = os.path.join(self.args.checkpoints, setting)\n",
    indent + "    candidates = ['0_checkpoint.pth','best_model.pth','checkpoint.pth','best.pth','last.pth']\n",
    indent + "    ckpt_path = None\n",
    indent + "    for name in candidates:\n",
    indent + "        pth = os.path.join(ckpt_dir, name)\n",
    indent + "        if os.path.exists(pth):\n",
    indent + "            ckpt_path = pth\n",
    indent + "            break\n",
    indent + "    if ckpt_path is None:\n",
    indent + "        print(f\"[FORCE_LOAD] no ckpt found in {ckpt_dir} (candidates={candidates}) -> using current weights\")\n",
    indent + "    else:\n",
    indent + "        print(f\"[FORCE_LOAD] loading ckpt: {ckpt_path}\")\n",
    indent + "        state = torch.load(ckpt_path, map_location=self.device)\n",
    indent + "        # 兼容: 可能保存的是 {'model':..., ...} 或直接是 state_dict\n",
    indent + "        if isinstance(state, dict) and ('state_dict' in state or 'model' in state):\n",
    indent + "            sd = state.get('state_dict', state.get('model'))\n",
    indent + "        else:\n",
    indent + "            sd = state\n",
    indent + "        self.model.load_state_dict(sd, strict=False)\n",
    indent + "except Exception as e:\n",
    indent + "    print(f\"[FORCE_LOAD] load failed: {e} -> using current weights\")\n",
    "\n"
]

s[body_idx:body_idx] = insert
p.write_text(''.join(s), encoding="utf-8")
print("OK: patched exp.test() to force-load checkpoint if present.")
