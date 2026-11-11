"""List every $...$ / $$...$$ equation in the given markdown files and check it converts.

usage: python build/check-math.py file.md [...]   (use the repo's build venv)
Exit code 1 if any equation fails to convert to MathML.
"""
import sys
from markdown_it import MarkdownIt
from mdit_py_plugins.dollarmath import dollarmath_plugin
import latex2mathml.converter

md = MarkdownIt('commonmark').use(dollarmath_plugin, allow_space=False, allow_digits=False, double_inline=True).enable('table')
failed = 0
for path in sys.argv[1:]:
    tokens = md.parse(open(path).read())
    stack = list(tokens)
    while stack:
        t = stack.pop(0)
        if t.children: stack[0:0] = t.children
        if t.type.startswith("math"):
            line = (t.map[0] + 1) if t.map else "?"
            try:
                latex2mathml.converter.convert(t.content)
                status = "ok"
            except Exception as e:
                status = f"FAIL {e}"; failed += 1
            print(f"{path}:{line}: [{t.type}] {status}: {t.content[:70]!r}")
sys.exit(1 if failed else 0)
