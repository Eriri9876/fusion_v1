from pathlib import Path

p = Path("main_MLF_longterm.py")
lines = p.read_text(encoding="utf-8", errors="ignore").splitlines(True)

marker = "# [PATCH] RESPECT_CLI_BEGIN"
if any(marker in ln for ln in lines):
    print("SKIP: patch already applied.")
    raise SystemExit(0)

# 找到 main() 内部缩进的 if args.is_training:
target_idx = None
indent = None
for i, ln in enumerate(lines):
    if ln.lstrip().startswith("if args.is_training"):
        leading = ln[:len(ln) - len(ln.lstrip())]
        if len(leading) > 0:
            target_idx = i
            indent = leading
            break

if target_idx is None:
    raise SystemExit("ERROR: cannot find an indented 'if args.is_training' inside main().")

# 注意：下面这行要在 main() 运行时再计算 args，所以这里用普通字符串拼接，不用外层 f-string 求值
patch_lines = [
    indent + marker + "\n",
    indent + "import sys as _sys\n",
    indent + "def _cli_value(flag, default=None):\n",
    indent + "    if flag in _sys.argv:\n",
    indent + "        try:\n",
    indent + "            return _sys.argv[_sys.argv.index(flag)+1]\n",
    indent + "        except Exception:\n",
    indent + "            return default\n",
    indent + "    return default\n",
    "\n",
    indent + "_v = _cli_value('--is_training', None)\n",
    indent + "if _v is not None:\n",
    indent + "    try:\n",
    indent + "        args.is_training = int(str(_v).strip())\n",
    indent + "    except Exception:\n",
    indent + "        pass\n",
    "\n",
    indent + "_ck = _cli_value('--checkpoints', None)\n",
    indent + "if _ck is not None:\n",
    indent + "    args.checkpoints = _ck\n",
    "\n",
    indent + "if hasattr(args, 'is_training') and int(getattr(args, 'is_training', 1)) == 0:\n",
    indent + "    if hasattr(args, 'state'):\n",
    indent + "        args.state = 'test'\n",
    "\n",
    indent + "print('[PATCH] effective is_training=%s checkpoints=%s' % (getattr(args,'is_training',None), getattr(args,'checkpoints',None)))\n",
    indent + "# [PATCH] RESPECT_CLI_END\n",
    "\n",
]

lines[target_idx:target_idx] = patch_lines
p.write_text("".join(lines), encoding="utf-8")
print("OK: inserted RESPECT_CLI patch safely.")
