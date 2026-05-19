# milk-cli Scripting Cheatsheet

## Variables

```
VAR=val              # String assign (no spaces)
VAR = expr           # Calculator assign (spaces!)
$VAR  ${VAR}         # Read variable
unset VAR            # Remove
vars                 # List all
export VAR=val       # Environment
readonly VAR=val     # Immutable
local VAR=val        # Function-local
$?                   # Last return value
```

## String Ops

```
${#v}        length       ${v:n:m}     substr
${v%%pat}    strip suffix ${v##pat}    strip prefix
${v/p/r}     replace 1st  ${v//p/r}    replace all
${v^^}       uppercase    ${v,,}       lowercase
${v:-def}    default      ${v:=def}    assign default
${v:+alt}    alt if set   ${v:?msg}    err if unset
${!v}        indirect
```

## Arrays

```
arr=(a b c)          ${arr[0]}  ${arr[@]}
${#arr[@]}           read -a arr
mapfile -t arr < f   declare -A map
map[k]=v             ${map[k]}
```

## Arithmetic

```
y=$(( x + 5 ))       # Integer math
(( x > 5 ))          # Conditional
let "x=1+2"          # Let
```

## Native Calculator (spaced =)

```
a = 2 + 3 * 4        # Scalar: 14
out = img1 + img2     # Image element-wise
mask = (img > 0.5)    # Boolean mask
s = itot(mask)        # Sum pixels
```

**Functions:** sin cos tan asin acos atan atan2
sinh cosh tanh exp log log2 log10 sqrt cbrt pow
floor ceil round trunc abs fabs min max fmod
where(cond, T, F)

**Image:** itot imean imin imax dot norm
imcrop

## Flow Control

```
if [ $x -gt 5 ]; then ... elif ...; then
    ... else ... fi
while [ $n -lt 10 ]; do ... done
until [ $done -eq 1 ]; do ... done
for i in a b c; do ... done
for i in {1..10}; do ... done
for ((i=0;i<10;i++)); do ... done
case $v in pat) cmd ;; *) cmd ;; esac
break [N]   continue [N]   return [val]
```

## Test Operators

```
Numeric: -eq -ne -gt -ge -lt -le
String:  = != -n -z
File:    -f -d -e -s -r -w -x -L
Milk:    -S (stream) -F (FPS) -P (proc) -v (var)
Regex:   [[ $s =~ ^pattern$ ]]
Negate:  [ ! expr ]
```

## Functions

```
function name { body }
name arg1 arg2          # $1..$9 in body
local v=val             # Scoped variable
shift [N]               # Rotate params
getopts "ab:" opt       # Option parsing
```

## FPS Access

```
@fpsname.param           # Read
@fpsname.param=value     # Write
fpsset fpsname param val # Write (command)
waitfor_fps name 10      # Wait for FPS
wait -F name p=v 5       # Wait param=val
```

## Stream Access

```
mem.mk2Dim name W H      # Create
mem.rm name 0             # Delete (0=ign err)
mem.listim                # List
@s.<stream>.xsize  @s.<stream>.ysize   # Dimensions
@s.<stream>.naxis  @s.<stream>.type    # Axes, datatype
@s.<stream>.cnt0   @s.<stream>.nelement# Counter, size
waitfor_stream name 10    # Wait for SHM
wait -S name 5            # Wait for update
on_update name { cmd }    # Trigger
```

## Process Control

```
procctl name run|pause|stop
procwait name ACTIVE 10
procstat name
@procname.pid  @procname.loopfreq
```

## I/O

```
echo [-n] args           printf "fmt" a..
read [-p prompt] var     read -t 5 var
cmd > file   cmd >> file cmd < file
read v <<< "str"         $(cmd)
cmd &                    wait
cmd1 | cmd2              !syscommand
```

## Utilities

```
sleep N    usleep N    time cmd
timeout N cmd          watch -n N cmd
source f   basename p  dirname p
seq S [I] E            pushd/popd/dirs
alias n='cmd'          true / false
```

## Error Handling

```
set -e    set -x    set +e    set +x
trap 'cleanup' EXIT INT TERM
```

## Chaining

```
cmd1 ; cmd2    cmd1 && cmd2    cmd1 || cmd2
```
