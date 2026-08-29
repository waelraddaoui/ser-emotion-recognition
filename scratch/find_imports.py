import os

exclude_dirs = {'.venv', '.venv311', '.idea', '.vscode', '__pycache__', 'scratch'}
targets = ['import tensorflow', 'from tensorflow', 'import keras', 'from keras']

for root, dirs, files in os.walk('.'):
    dirs[:] = [d for d in dirs if d not in exclude_dirs]
    for file in files:
        if file.endswith('.py'):
            path = os.path.join(root, file)
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    for i, line in enumerate(f):
                        if any(t in line for t in targets):
                            print(f"{path}:{i+1}: {line.strip()}")
            except Exception as e:
                print(f"Error reading {path}: {e}")
