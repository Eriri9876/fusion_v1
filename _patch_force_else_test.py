from pathlib import Path
import re

p = Path("main_MLF_longterm.py")
lines = p.read_text(encoding="utf-8", errors="ignore").splitlines(True)

marker = "# [PATCH] FORCE_TEST_ELSE"
if any(marker in ln for ln in lines):
    print("SKIP: FORCE_TEST_ELSE already applied.")
    raise SystemExit(0)

# 找到 main() 内缩进的 if args.is_training:
if_idx = None
indent = None
for i, ln in enumerate(lines):
    if ln.lstrip().startswith("if args.is_training"):
        leading = ln[:len(ln)-len(ln.lstrip())]
        if len(leading) > 0:
            if_idx = i
            indent = leading
            break
if if_idx is None:
    raise SystemExit("ERROR: cannot find indented 'if args.is_training' inside main().")

# 如果紧跟着已经有 else:，就不处理
for j in range(if_idx+1, min(if_idx+120, len(lines))):
    if lines[j].lstrip().startswith("else:") and lines[j].startswith(indent):
        print("INFO: else branch already exists near if args.is_training. Not inserting.")
        raise SystemExit(0)

# 找到训练块里 exp.train(setting) 的行，用它作为插入点（插在训练块之后）
train_call = None
for j in range(if_idx, min(if_idx+250, len(lines))):
    if "exp.train(" in lines[j].replace(" ", ""):
        train_call = j
        break
if train_call is None:
    raise SystemExit("ERROR: cannot find 'exp.train(' after if args.is_training. File structure unexpected.")

# 找到训练块结束：向后找缩进回到 indent 层级（或文件结束）
insert_at = None
for k in range(train_call+1, min(train_call+400, len(lines))):
    # 空行跳过
    if lines[k].strip() == "":
        continue
    lead = lines[k][:len(lines[k]) - len(lines[k].lstrip())]
    # 回到和 if 同层级或更外层，说明训练块结束
    if len(lead) <= len(indent) and not lines[k].lstrip().startswith("#"):
        insert_at = k
        break
if insert_at is None:
    insert_at = len(lines)

sub = indent + "    "
else_block = [
    indent + "else:\n",
    sub + marker + "\n",
    sub + "for ii in range(args.itr):\n",
    sub + "    setting = f'{args.data_real}_{args.model}_{\"MultiPeriod_128_384_512_768_1024\"}_pl{args.pred_len}'\n",
    sub + "    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))\n",
    sub + "    exp.test(setting)\n",
    "\n",
]

lines[insert_at:insert_at] = else_block
p.write_text("".join(lines), encoding="utf-8")
print("OK: inserted else: exp.test(setting) branch.")
