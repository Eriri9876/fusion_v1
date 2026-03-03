from pathlib import Path
import re

p = Path(r"exp/exp_main_public.py")
txt = p.read_text(encoding="utf-8", errors="ignore").splitlines(True)

marker = "# [PATCH] MKDIR_BEFORE_FINAL_TEST"
if any(marker in ln for ln in txt):
    print("SKIP: patch already applied.")
    raise SystemExit(0)

# 找到写 final_test.json 的位置
target = None
for i, ln in enumerate(txt):
    if "final_test" in ln and "open(" in ln and ".json" in ln:
        target = i
        break

if target is None:
    raise SystemExit("ERROR: cannot find line that opens final_test.json")

# 取缩进
indent = txt[target][:len(txt[target]) - len(txt[target].lstrip())]

# 在 open 前插入 makedirs
insert = [
    indent + marker + "\n",
    indent + "import os\n" if not any(re.match(r"^\s*import os\s*$", x) for x in txt[:80]) else "",
    indent + "os.makedirs(path, exist_ok=True)\n",
]

# 过滤空字符串
insert = [x for x in insert if x != ""]

txt[target:target] = insert
p.write_text("".join(txt), encoding="utf-8")
print("OK: inserted os.makedirs(path, exist_ok=True) before writing final_test.json.")
