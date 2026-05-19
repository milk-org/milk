import os
import sys
import re

def remove_strings_and_comments(s):
    s = re.sub(r'".*?(?<!\\)"', '""', s)
    s = re.sub(r"'.*?(?<!\\)'", "''", s)
    s = re.sub(r'//.*', '', s)
    s = re.sub(r'/\*.*?\*/', '', s)
    return s

def has_comment(s):
    clean_str = re.sub(r'".*?(?<!\\)"', '""', s)
    clean_str = re.sub(r"'.*?(?<!\\)'", "''", clean_str)
    return '//' in clean_str or '/*' in clean_str or '*/' in clean_str

def merge_file(filepath):
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        return False
        
    out_lines = []
    i = 0
    brace_depth = 0
    in_block_comment = False
    in_macro = False
    
    while i < len(lines):
        line = lines[i]
        s = line.strip()
        
        clean_line = remove_strings_and_comments(line)
        clean_s = clean_line.strip()
        
        # Macro tracking
        was_in_macro = in_macro
        if s.startswith('#'):
            was_in_macro = True
            
        if was_in_macro or s.startswith('#'):
            in_macro = s.endswith('\\')
        else:
            in_macro = False
            
        # Comment tracking
        if '/*' in clean_line:
            in_block_comment = True
            
        prevent_merge = (
            was_in_macro or 
            s.startswith('#') or 
            in_block_comment or 
            has_comment(line) or 
            s.endswith('\\') or
            not s or
            '{' in clean_s or
            '}' in clean_s
        )
        
        if not prevent_merge:
            if not clean_s.endswith(';') and not clean_s.endswith('{') and not clean_s.endswith('}'):
                buffer = [line]
                j = i + 1
                can_merge = False
                
                # Check ahead
                while j < len(lines):
                    ns = lines[j].strip()
                    nsclean = remove_strings_and_comments(lines[j]).strip()
                    
                    if has_comment(lines[j]) or ns.startswith('#') or ns.endswith('\\'):
                        break
                        
                    if '{' in nsclean or '}' in nsclean:
                        break
                        
                    buffer.append(lines[j])
                    if nsclean.endswith(';'):
                        can_merge = True
                        break
                        
                    j += 1
                    
                if can_merge:
                    leading_ws = line[:len(line) - len(line.lstrip())]
                    parts = [l.strip() for l in buffer]
                    merged = " ".join(parts)
                    
                    merged = merged.replace("( ", "(").replace("[ ", "[").replace(" )", ")").replace(" ]", "]").replace(" ,", ",")
                    merged = leading_ws + merged + "\n"
                    
                    if len(merged) <= 100:
                        clean_merged = remove_strings_and_comments(merged).strip()
                        
                        # At global scope, don't merge function prototypes or function-like macro calls
                        if brace_depth == 0 and clean_merged.endswith(');') and '=' not in clean_merged:
                            can_merge = False
                            
                        if can_merge:
                            brace_depth += clean_merged.count('{') - clean_merged.count('}')
                            out_lines.append(merged)
                            i = j + 1
                            if '*/' in clean_line:
                                in_block_comment = False
                            continue
        
        if '*/' in clean_line:
            in_block_comment = False
            
        if not in_block_comment and not was_in_macro and not s.startswith('#'):
            brace_depth += clean_line.count('{') - clean_line.count('}')
            
        out_lines.append(line)
        i += 1
        
    if lines != out_lines:
        with open(filepath, 'w') as f:
            f.writelines(out_lines)
        return True
    return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python merge_splits.py <file1> [<file2> ...]")
        sys.exit(0)
        
    changed = 0
    for target in sys.argv[1:]:
        if os.path.isfile(target):
            if target.endswith(('.c', '.h', '.cpp', '.hpp')):
                if merge_file(target):
                    print(f"Merged lines in {target}")
                    changed += 1
        elif os.path.isdir(target):
            for root, dirs, files in os.walk(target):
                for f in files:
                    if f.endswith(('.c', '.h', '.cpp', '.hpp')):
                        filepath = os.path.join(root, f)
                        if merge_file(filepath):
                            print(f"Merged lines in {filepath}")
                            changed += 1
                    
    if changed > 0:
        print(f"Total files formatted: {changed}")
        sys.exit(1)
    else:
        sys.exit(0)
