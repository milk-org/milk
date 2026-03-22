import re
import os

with open("_build/warnings.log", "r") as f:
    lines = f.readlines()

warnings = []
for line in lines:
    # Example: /path/to/file.c:130:25: warning: ‘compute_function’ defined but not used [-Wunused-function]
    m = re.match(r'^(/home/oguyon[^\:]+):(\d+):(\d+)?:\s*warning:\s*(.*)$', line)
    if m:
        path = m.group(1)
        lineno = int(m.group(2))
        msg = m.group(4)
        warnings.append((path, lineno, msg))

# group by file
files = {}
for w in warnings:
    path, lineno, msg = w
    if path not in files:
        files[path] = []
    files[path].append((lineno, msg))

for path in files.keys():
    if not os.path.exists(path):
        continue
    with open(path, "r") as f:
        content = f.read().split('\n')
    
    modifications = 0
    
    # process in reverse to avoid shifting line numbers if we ever add lines
    # but currently we just replace inline
    for lineno, msg in files[path]:
        idx = lineno - 1
        if idx >= len(content):
            continue
            
        line = content[idx]
        
        if "set but not used" in msg or "unused variable" in msg or "unused parameter" in msg or "defined but not used" in msg:
            if "unused parameter" in msg or "unused variable" in msg or "set but not used" in msg:
                # find variable name
                m = re.search(r"‘([^’]+)’", msg)
                if m:
                    var = m.group(1)
                    # if the line contains the variable
                    if var in line and "__attribute__((unused))" not in line:
                        # find the variable declaration and append attribute
                        new_line = re.sub(r'\b' + re.escape(var) + r'\b(?![\w_])', var + ' __attribute__((unused))', line, count=1)
                        if new_line != line:
                            content[idx] = new_line
                            modifications += 1
            elif "defined but not used" in msg:
                # it's usually a static function or variable
                m = re.search(r"‘([^’]+)’", msg)
                if m:
                    name = m.group(1)
                    if name in line and "__attribute__((unused))" not in line:
                        # For functions, we might just put __attribute__((unused)) before the name
                        new_line = re.sub(r'\b' + re.escape(name) + r'\b(?![\w_])', '__attribute__((unused)) ' + name, line, count=1)
                        if new_line != line:
                            content[idx] = new_line
                            modifications += 1
        elif "ignoring ‘#pragma omp simd’" in msg or "ignoring ‘#pragma omp parallel’" in msg:
            if "#pragma omp" in line or (idx-1 >= 0 and "#pragma omp" in content[idx-1]):
                content[idx] = "// " + content[idx]
                modifications += 1
    if modifications > 0:
        with open(path, "w") as f:
            f.write("\n".join(content))

print("Done running script.")
