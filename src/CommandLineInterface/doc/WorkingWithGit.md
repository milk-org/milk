# Working with git {#page_WorkingWithGit}

@note This file: ./src/CommandLineInterface/doc/WorkingWithGit.md

## Modules

Most of milk's code is in a set of modules. Each module is compiled as a shared object loaded by the main process at runtime.

In Git, each module has its own repository, linked to the package as a git submodule.




## Synchronization: loading updated modules

To get latest modules (master branches) :

	git submodule foreach "(git checkout master; git pull)"

To get latest modules (dev branches) :

	git submodule foreach "(git checkout dev; git pull)"


## Editing submodule source code

When developing, work in dev branch and set branch to dev in all submodules and main package:

	git submodule foreach "(git checkout dev; git pull; git push)"
	git checkout dev
	git pull
	git push
	

## Updating master branches

To synchronize master to latest dev:
	
	git submodule foreach "(git checkout master; git merge dev)"
	git checkout master
	git merge dev
	git submodule foreach "(git checkout master; git push)"
	git push

When updating master, it's good practice to also issue a git tag

	git tag -a v0.1.23 -m "Version 0.1.23"
	git push origin v0.1.23




## Documentation

Documentation tree can be locally built on dev branch with doxygen from main directory and pushed by :

	git checkout dev
	doxygen
	git add html
	git commit -am 'updated doxygen documentation'
	git push

Same should be done on branch master.

When switching branches, you may get an error message "The following untracked working tree files would be overwritten by checkout" preventing branch switch. If the error only lists files in the html directory, you can safely use the -f option to force branch change.


