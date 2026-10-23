#!/bin/bash
# CTest milkctrl_keys_doc: the key tables in docs/guide/milk-CTRL.md are what `milk-CTRL --keys-markdown` prints.
# Usage: check_keys_doc.sh MILK_CTRL GUIDE_MD
set -u
if diff -u <(sed -n '/<!-- keys:begin -->/,/<!-- keys:end -->/{//!p}' "$2") <("$1" --keys-markdown); then
    exit 0
fi
echo "$2: key tables out of date; regenerate them with: $1 --keys-markdown" >&2
exit 1
