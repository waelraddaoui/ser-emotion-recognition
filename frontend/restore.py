import json
import re

def restore():
    log_path = r'C:\Users\marie\.gemini\antigravity\brain\72712a38-f586-499e-bad6-363999cfbd0b\.system_generated\logs\transcript.jsonl'
    
    ser_content = None
    admin_content = None
    
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                if data.get('type') == 'TOOL_RESPONSE':
                    out = data.get('content', '')
                    if not ser_content and 'ser.component.html`' in out and 'Total Lines:' in out:
                        # Parse
                        lines = out.split('\n')
                        content_lines = []
                        for l in lines:
                            if re.match(r'^\d+: ', l):
                                content_lines.append(l.split(': ', 1)[1])
                        if len(content_lines) > 250:
                            ser_content = '\n'.join(content_lines)
                            print(f"Found ser.component.html with {len(content_lines)} lines")
                    
                    if not admin_content and 'admin-dashboard.component.html`' in out and 'Total Lines:' in out:
                        lines = out.split('\n')
                        content_lines = []
                        for l in lines:
                            if re.match(r'^\d+: ', l):
                                content_lines.append(l.split(': ', 1)[1])
                        if len(content_lines) > 550:
                            admin_content = '\n'.join(content_lines)
                            print(f"Found admin-dashboard.component.html with {len(content_lines)} lines")
            except Exception as e:
                pass

    if ser_content:
        with open(r'c:\Users\marie\Desktop\Projets\PFE Issat\frontend\src\app\ser\ser.component.html', 'w', encoding='utf-8') as f:
            f.write(ser_content)
        print('Restored ser.component.html')
    else:
        print('Could not find ser.component.html > 200 lines')

    if admin_content:
        with open(r'c:\Users\marie\Desktop\Projets\PFE Issat\frontend\src\app\admin\admin-dashboard.component.html', 'w', encoding='utf-8') as f:
            f.write(admin_content)
        print('Restored admin-dashboard.component.html')
    else:
        print('Could not find admin-dashboard.component.html > 550 lines')

if __name__ == '__main__':
    restore()
