# ztime

A simple decorator for timing functions.

## Installation

```bash
pip install ztime
```

## Usage

```python
from ztime import ztime
import time

@ztime
def my_function():
    time.sleep(1)

@ztime(repeats=3)
def another_function():
    time.sleep(0.5)

my_function()
another_function()
```
