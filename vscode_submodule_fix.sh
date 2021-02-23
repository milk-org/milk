#!/usr/bin/env bash

git checkout -- .gitmodules

for mod in `ls -c1 -d ./plugins/*/*`
do
cat >> .gitmodules << EOF
[submodule "${mod}"]
	path = ${mod}
	url = ""
EOF
done
