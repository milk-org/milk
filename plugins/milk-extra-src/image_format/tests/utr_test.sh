#!/bin/bash

milk << EOF
mload milkimageformat
readshmim a
#readshmim b
#readshmim c
.cred_ql_utr ..procinfo 1
imgformat.cred_ql_utr ..triggermode 1
#imgformat.cred_ql_utr ..triggersname "a"
#imgformat.cred_ql_utr ..triggerdelay 0.001
imgformat.cred_ql_utr ..loopcntMax -1
imgformat.cred_ql_utr a b c 5000
listim
exitCLI
EOF
