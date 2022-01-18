#!/bin/bash

if [[ ! -z "${MILK_ROOT}" ]]; then
    echo "Changing dir to ${MILK_ROOT}"
    cd ${MILK_ROOT}
fi

pip install pre-commit

sudo apt install shellcheck

pre-commit install


for FOLD in $(ls -d ${MILK_ROOT}/plugins/milk-extra-src/*/); do
    echo "Installing pre-commit in ${FOLD}"
    cp ${MILK_ROOT}/.style.yapf ${FOLD}/
    cp ${MILK_ROOT}/.pre-commit-config.yaml ${FOLD}/
    cp ${MILK_ROOT}/.clang-format ${FOLD}/
    cd ${FOLD}
    pre-commit install
    cd ${MILK_ROOT}
done

FOLD=${MILK_ROOT}/plugins/cacao-src/
echo "Installing pre-commit in ${FOLD}"
cp ${MILK_ROOT}/.style.yapf ${FOLD}/
cp ${MILK_ROOT}/.pre-commit-config.yaml ${FOLD}/
cp ${MILK_ROOT}/.clang-format ${FOLD}/
cd ${FOLD}
pre-commit install
cd ${MILK_ROOT}

FOLD=${MILK_ROOT}/plugins/coffee-src/
echo "Installing pre-commit in ${FOLD}"
cp ${MILK_ROOT}/.style.yapf ${FOLD}/
cp ${MILK_ROOT}/.pre-commit-config.yaml ${FOLD}/
cp ${MILK_ROOT}/.clang-format ${FOLD}/
cd ${FOLD}
pre-commit install
cd ${MILK_ROOT}

echo "Now you may run \"pre-commit run --all-files\" here and in plugin repos"
