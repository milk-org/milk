---
description: Audit switch blocks for incomplete data type handling
---

# Check Type Consistency

Run this workflow to find `switch` statements that
handle image/stream data types but are missing
cases for some types. This catches silent data
corruption or unsupported-type crashes.

## 1. Find All Type Switch Blocks

Search for switch statements on data type variables:

```bash
grep -rn 'switch.*datatype\|switch.*atype\|case _DATATYPE_\|case CLIARG_' \
  src/ plugins/ --include='*.c' | head -80
```

## 2. Define the Complete Type Set

The canonical set of image data types is:

```
_DATATYPE_UINT8, _DATATYPE_INT8,
_DATATYPE_UINT16, _DATATYPE_INT16,
_DATATYPE_UINT32, _DATATYPE_INT32,
_DATATYPE_UINT64, _DATATYPE_INT64,
_DATATYPE_FLOAT, _DATATYPE_DOUBLE,
_DATATYPE_COMPLEX_FLOAT,
_DATATYPE_COMPLEX_DOUBLE
```

## 3. Audit Each Switch Block

For each `switch` block found:

1. List all `case` labels present.
2. Compare against the canonical set.
3. Flag any missing types.
4. Determine if the omission is intentional
   (e.g., complex types not applicable) or a bug.

## 4. Categorize Findings

Classify each incomplete switch as:

- **Bug**: A type that should be handled but isn't
  (e.g., `INT8` missing where all other integer
  types are handled).
- **Intentional**: A type that doesn't apply
  (e.g., complex types in a pixel-coordinate
  function). These should have a `default:` case
  with an error message.

## 5. Fix or Report

- For bugs: add the missing case, following the
  pattern of existing cases in that function.
- For intentional omissions: ensure there is a
  `default:` case that prints a clear error.
- Report the total number of switch blocks audited,
  bugs found, and fixes applied.
