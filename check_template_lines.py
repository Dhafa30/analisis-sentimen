import sys
import re

with open(r"d:\tst\analyzer\templates\index.html", "r", encoding="utf-8") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    tags = re.findall(r'{%.*?%}', line)
    if tags:
        print(f"Line {i+1}: {tags}")
