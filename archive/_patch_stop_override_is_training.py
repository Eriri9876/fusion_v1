import re
from pathlib import Path

p = Path("main_MLF_longterm.py")
s = p.read_text(encoding="utf-8", errors="ignore")

# 1) 注释掉任何强行把 is_training 改成 True 的语句
s2 = re.sub(r"(?m)^\s*args\.is_training\s*=\s*True\s*$",
            r"# [PATCHED] args.is_training=True (do not override CLI)",
            s)

# 2) 如果有人把 state 写死 train，也注释掉
s2 = re.sub(r"(?m)^\s*args\.state\s*=\s*['\"]train['\"]\s*$",
            r"# [PATCHED] args.state='train' (do not override CLI)",
            s2)

if s2 == s:
    print("WARN: no forced lines found (maybe already clean).")
else:
    p.write_text(s2, encoding="utf-8")
    print("OK: patched main_MLF_longterm.py (stop overriding CLI is_training/state).")
