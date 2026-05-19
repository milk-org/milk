#!/bin/bash
find _build -name "*.o" > obj_files.txt

echo "Finding defined symbols..."
xargs nm -g --defined-only < obj_files.txt | awk '{if (NF==3 && $2=="T") print $3}' | sort | uniq > defined.txt

echo "Finding undefined symbols..."
xargs nm -g -u < obj_files.txt | awk '{if (NF==2 && $1=="U") print $2}' | sort | uniq > undefined.txt

echo "Finding potentially unused exported functions..."
comm -23 defined.txt undefined.txt > unused_global_functions.txt

echo "Total defined:" $(wc -l < defined.txt)
echo "Total undefined:" $(wc -l < undefined.txt)
echo "Total potentially unused:" $(wc -l < unused_global_functions.txt)
head -n 20 unused_global_functions.txt
