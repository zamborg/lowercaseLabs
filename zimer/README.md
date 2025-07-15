# ztime

A simple decorator for timing functions.

## Installation

```bash
pip install zimer
```

## Usage

```python
from zimer import zimer
import time

@zimer
def my_function():
    time.sleep(1)

@zimer(repeats=3)
def another_function():
    time.sleep(0.5)

my_function()
another_function()
```
