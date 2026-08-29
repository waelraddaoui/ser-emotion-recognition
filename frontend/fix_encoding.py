import os
import glob

def fix_encoding(filepath):
    try:
        with open(filepath, 'r', encoding='windows-1252') as f:
            content = f.read()
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Fixed encoding for {filepath}")
    except Exception as e:
        print(f"Failed {filepath}: {e}")

files = [
    r'src\app\ser\ser.component.html',
    r'src\app\admin\admin-dashboard.component.html'
]

for file in files:
    fix_encoding(file)
