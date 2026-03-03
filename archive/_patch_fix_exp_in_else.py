from pathlib import Path
import re

p = Path("main_MLF_longterm.py")
lines = p.read_text(encoding="utf-8", errors="ignore").splitlines(True)

marker = "# [PATCH] FORCE_TEST_ELSE"
idx = None
for i, ln in enumerate(lines):
    if marker in ln:
        idx = i
        break
if idx is None:
    raise SystemExit("ERROR: cannot find FORCE_TEST_ELSE marker. Did you apply the else-test patch?")

# else 分支里，找到 for ii in range(args.itr) 那行，在它前面插入 exp = Exp(args)
inserted = False
for j in range(idx, min(idx + 40, len(lines))):
    if "exp = Exp(args)" in lines[j].replace(" ", ""):
        print("SKIP: exp already created in else branch.")
        raise SystemExit(0)
    if lines[j].lstrip().startswith("for ii in range(args.itr)"):
        indent = lines[j][:len(lines[j]) - len(lines[j].lstrip())]
        lines.insert(j, indent + "exp = Exp(args)\n")
        inserted = True
        break

if not inserted:
    raise SystemExit("ERROR: cannot find 'for ii in range(args.itr)' near else branch to insert exp initialization.")

p.write_text("".join(lines), encoding="utf-8")
print("OK: inserted `exp = Exp(args)` into else(test) branch.")
