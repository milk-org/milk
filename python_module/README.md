# CacaoProcessTools

- [CacaoProcessTools](#cacaoprocesstools)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Streams: milk-CTRL](#streams-milk-ctrl)
  - [Open processCTRL](#open-processctrl)

## Installation

In the root directory ie. `processtools`

```bash
pip install .
```

Note: You can install it in a custom path with the option `-t $HOME/local/python`

Note2: If you update it, use the option `-U`

## Usage

```python
import CacaoProcessTools as CPT
```

## Streams: milk-CTRL

The stream monitor is the F3 view of milk-CTRL (`milk-streamCTRL` opens it there); start it as a program:

```python
>>> import subprocess
>>> subprocess.run(["milk-streamCTRL"])
```

## Open processCTRL

```python
>>> CPT.processCTRL()
```
