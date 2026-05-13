import subprocess
import os
import sys

# Find all .o files in cacao-src (ignoring _compute and fpsexec)
output = subprocess.check_output('find ../plugins/cacao-src -name "*.o" ! -path "*/CMakeFiles/cacao-*fpsexec*" ! -path "*/CMakeFiles/*_compute*" -type f', shell=True)
o_files = output.decode().strip().split('\n')

defs = {}
refs = set()

for o_file in o_files:
    if not o_file: continue
    nm_out = subprocess.check_output(['nm', '-g', o_file]).decode().split('\n')
    for line in nm_out:
        parts = line.strip().split()
        if len(parts) >= 2 and parts[-2] in ['T', 'D', 'B', 'R', 'C']:
            sym = parts[-1]
            if sym not in defs:
                defs[sym] = []
            defs[sym].append(o_file)
        elif len(parts) >= 2 and parts[-2] == 'U':
            refs.add(parts[-1])

# Also check core milk .o files to ensure we don't mark cacao things static that are used by milk
output_milk = subprocess.check_output('find .. -name "*.o" ! -path "*/cacao-src/*" -type f', shell=True)
milk_o_files = output_milk.decode().strip().split('\n')
for o_file in milk_o_files:
    if not o_file: continue
    try:
        nm_out = subprocess.check_output(['nm', '-g', o_file]).decode().split('\n')
        for line in nm_out:
            parts = line.strip().split()
            if len(parts) >= 2 and parts[-2] == 'U':
                refs.add(parts[-1])
    except:
        pass

static_candidates = []
for sym, locations in defs.items():
    if sym not in refs:
        # Check if it's an init_ module function, which are looked up via dlsym
        if sym.startswith("init_"):
            continue
        # CLIcmd functions are exported to be added via CLIADDCMD in the same file usually? Wait.
        # Actually, if they are only used in the same file, they should be static!
        # Let's collect them
        for loc in locations:
            # Get the source file name instead of object file
            src_file = os.path.basename(loc).replace('.o', '')
            static_candidates.append(f"{sym} in {loc}")

print(f"Found {len(static_candidates)} candidates")
with open("candidates.txt", "w") as f:
    for c in sorted(static_candidates):
        f.write(c + "\n")
