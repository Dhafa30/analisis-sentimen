import os
import django
from django.conf import settings
from django.template import Template, Context

settings.configure(
    TEMPLATES=[{
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'APP_DIRS': True,
    }]
)
django.setup()

with open(r"d:\tst\analyzer\templates\index.html", "r", encoding="utf-8") as f:
    content = f.read()

try:
    Template(content)
    print("Template parsed successfully!")
except Exception as e:
    import traceback
    traceback.print_exc()
