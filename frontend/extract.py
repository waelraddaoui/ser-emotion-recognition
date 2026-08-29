import json

def extract():
    log_path = r'C:\Users\marie\.gemini\antigravity\brain\72712a38-f586-499e-bad6-363999cfbd0b\.system_generated\logs\transcript.jsonl'
    matches = []
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if 'ser.component.html`' in line and 'Total Lines:' in line:
                    matches.append((i, line))
    except Exception as e:
        print(e)
    
    with open('matches.txt', 'w', encoding='utf-8') as out:
        for i, m in matches:
            out.write(f"Line {i}: Length {len(m)}\n")
            # Write a snippet
            data = json.loads(m)
            content = data.get('content', '')
            out.write(content[:1000] + '\n\n')

if __name__ == '__main__':
    extract()
