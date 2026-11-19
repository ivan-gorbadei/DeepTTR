"""The commands of README.md's "Play against it" section, so that CI runs them as a reader types
them.

usage: python .github/scripts/readme_block.py powershell|sh install|run

`install` prints every command of the section's block for that shell but the last, `run` prints
the last one, the command that starts the play server. With README_CLONE_FROM set, `git clone`
clones that path instead of the repository on GitHub, so that CI tests the commit it checked out.
"""
import os
import re
import sys
from pathlib import Path

CLONE = re.compile(r"^git clone (https://github\.com/\S+)$")

shell, part = sys.argv[1], sys.argv[2]
readme = Path(__file__).resolve().parents[2] / "README.md"
section = readme.read_text(encoding="utf-8").split("## Play against it\n", 1)[1].split("\n## ", 1)[0]
blocks = re.findall(rf"^```{shell}\n(.*?)^```", section, re.S | re.M)
lines = [line for line in blocks[0].splitlines() if line.strip()] if blocks else []
if not lines or not CLONE.match(lines[0]) or "deepttr.serve" not in lines[-1]:
    sys.exit(f"README.md: the {shell} block of 'Play against it' is not a clone, an install and the server")
if os.environ.get("README_CLONE_FROM"):
    lines[0] = CLONE.sub(lambda m: f'git clone "{os.environ["README_CLONE_FROM"]}" DeepTTR', lines[0])
print("\n".join(lines[:-1] if part == "install" else lines[-1:]))
