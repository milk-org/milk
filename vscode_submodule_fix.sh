#!/usr/bin/env bash

git co -- .gitmodules

for mod in `ls -d extra/*`
do
cat >> .gitmodules << EOF
[submodule "${mod}"]
	path = ${mod}
	url = ""
EOF
done
