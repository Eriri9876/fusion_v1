from pathlib import Path

p = Path("main_MLF_longterm.py")
s = p.read_text(encoding="utf-8", errors="ignore")

marker = "# [PATCH] RESPECT_CLI_BEGIN"
if marker in s:
    print("SKIP: patch already applied.")
    raise SystemExit(0)

needle = "if args.is_training"
idx = s.find(needle)
if idx == -1:
    print("ERROR: cannot find 'if args.is_training' in main_MLF_longterm.py")
    raise SystemExit(1)

patch = r'''
# [PATCH] RESPECT_CLI_BEGIN
import sys as _sys
def _cli_value(flag, default=None):
    if flag in _sys.argv:
        try:
            return _sys.argv[_sys.argv.index(flag)+1]
        except Exception:
            return default
    return default

# re-apply CLI values AFTER any internal overrides
_v = _cli_value("--is_training", None)
if _v is not None:
    try:
        args.is_training = int(str(_v).strip())
    except Exception:
        pass

_ck = _cli_value("--checkpoints", None)
if _ck is not None:
    args.checkpoints = _ck

# optional: align only_test/state if code uses them
if not hasattr(args, "only_test"):
    args.only_test = False
if args.is_training == 0:
    args.only_test = True
    if hasattr(args, "state"):
        args.state = "test"

print(f"[PATCH] effective is_training={getattr(args,'is_training',None)} only_test={getattr(args,'only_test',None)} checkpoints={getattr(args,'checkpoints',None)}")
# [PATCH] RESPECT_CLI_END
'''

s2 = s[:idx] + patch + s[idx:]
p.write_text(s2, encoding="utf-8")
print("OK: inserted RESPECT_CLI patch before training/test branch.")
