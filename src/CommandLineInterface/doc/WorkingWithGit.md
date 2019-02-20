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


## Releasing a new version

In dev branch:
- Update version number in the CMakeList.txt
- Edit version number in README.md

Update master branch to current dev branch.

Issue a git tag for the version:

	git tag -a vX.YY.ZZ -m "milk version X.YY.ZZ"
	git push origin vX.YY.ZZ

Issue tags for submodules (optional, but helpful to track which submodule version makes it into package version):

	git submodule foreach git tag -a milk-vX.Y.ZZ -m "milk version X.YY.ZZ"
	git submodule foreach git push origin milk-vX.Y.ZZ

@note Modules inherit version numbers from the package(s) to which they belong. Consequently, modules that are shared between packages can have parallel version number histories. For example, the version history for a module may be: milk-v0.1.01 -> cacao-v0.1.01 -> cacao-v0.1.02 -> cacao-v0.2.00 -> milk-v0.1.02. Any new version, regardless of which package it is associated with, still includes all previous changes: there is a single version thread.





## Documentation

Documentation tree can be locally built on dev branch with doxygen from main directory  :

	git checkout dev
	doxygen

To push it on the origin :	

	cd dochtml/html
	git add *
	git commit -am 'updated doxygen documentation'
	git push


@note When switching branches, you may get an error message "The following untracked working tree files would be overwritten by checkout" preventing branch switch. If the error only lists files in the html directory, you can safely use the -f option to force branch change.


