#!/usr/bin/env bash

git checkout -- .gitmodules

for mod in `find plugins/milk-extra-src -mindepth 1 -maxdepth 1 -type d| grep -v .git`
do
    cat >> .gitmodules << EOF
[submodule "${mod}"]
	path = ${mod}
	url = ""
EOF
done

if [ -d "plugins/cacao-src" ]; then
    cat >> .gitmodules << EOF
[submodule "cacao"]
	path = plugins/cacao-src
	url = ""
EOF
fi

if [ -d "plugins/coffee-src" ]; then
    cat >> .gitmodules << EOF
[submodule "coffee"]
	path = plugins/coffee-src
	url = ""
EOF
fi
