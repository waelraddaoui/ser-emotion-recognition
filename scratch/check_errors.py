import os
import py_compile
import traceback

def check_syntax():
    exclude_dirs = {'.venv', '.venv311', '.idea', '.vscode', '__pycache__', 'scratch'}
    errors = []
    
    for root, dirs, files in os.walk('.'):
        # Modify dirs in-place to prune excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    py_compile.compile(file_path, doraise=True)
                except py_compile.PyCompileError as e:
                    errors.append((file_path, str(e)))
                except Exception as e:
                    errors.append((file_path, f"Unexpected error: {traceback.format_exc()}"))
                    
    return errors

if __name__ == '__main__':
    print("Checking python files for syntax errors...")
    errors = check_syntax()
    if errors:
        print(f"Found {len(errors)} syntax errors:")
        for path, err in errors:
            print(f"\n--- {path} ---")
            print(err)
    else:
        print("No syntax errors found in any python files.")
