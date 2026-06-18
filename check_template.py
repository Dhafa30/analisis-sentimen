import sys

with open(r"d:\tst\analyzer\templates\index.html", "r", encoding="utf-8") as f:
    content = f.read()

import re
tags = re.findall(r'{%\s*(if|elif|else|endif).*?%}', content)
print("Tags found:", len(tags))

stack = []
for tag in tags:
    if tag == 'if':
        stack.append('if')
    elif tag == 'elif':
        pass
    elif tag == 'else':
        pass
    elif tag == 'endif':
        if not stack:
            print("ERROR: Unmatched endif")
        else:
            stack.pop()

print("Remaining in stack:", len(stack))

