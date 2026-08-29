import os
import sys

# Reconfigure stdout to utf-8 if possible
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def read_file_safely(path):
    encodings = ['utf-8', 'utf-16', 'utf-16le', 'utf-16be', 'latin-1', 'cp1252']
    for enc in encodings:
        try:
            with open(path, 'r', encoding=enc) as f:
                content = f.read()
                return enc, content
        except Exception:
            continue
    return None, None

def safe_print(text):
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode(sys.stdout.encoding or 'ascii', errors='replace').decode(sys.stdout.encoding or 'ascii'))

log_files = [
    'logs/admin_training.log',
    'logs/backend.err.log',
    'logs/backend.out.log',
    'logs/frontend.err.log',
    'logs/frontend.out.log',
    'train_8c_output.log',
    'train_output.txt',
    'out.log'
]

for lf in log_files:
    if os.path.exists(lf):
        enc, content = read_file_safely(lf)
        if content is not None:
            lines = content.splitlines()
            safe_print(f"\n==================================================")
            safe_print(f"FILE: {lf} (detected encoding: {enc})")
            safe_print(f"Total lines: {len(lines)}")
            safe_print(f"==================================================")
            if len(lines) <= 40:
                for line in lines:
                    safe_print(line)
            else:
                for line in lines[:20]:
                    safe_print(line)
                safe_print("... [TRUNCATED] ...")
                for line in lines[-20:]:
                    safe_print(line)
        else:
            safe_print(f"\nCould not read {lf} with any safe encoding.")
    else:
        safe_print(f"\nFile {lf} does not exist.")
