# Slugify Reduction Task

## Map Task

Create a Python module `slugify.py` that exposes:

```python
def slugify(text: str) -> str:
    ...
```

Requirements:

- lowercase the string
- trim leading and trailing whitespace
- replace any run of whitespace with a single `-`
- keep existing hyphens
- remove characters other than lowercase letters, digits, spaces, and hyphens
- collapse repeated hyphens to a single `-`
- strip leading and trailing hyphens

You must:

- run `python test_slugify.py`
- ensure the tests pass
- write `RESULT.md` that includes:
  - whether tests passed
  - the line count of `slugify.py`
  - a short note about the approach

## Reduce Task

Inspect all mapper workspaces and evaluate their `slugify.py` implementations.

Selection rule:

1. Only consider candidates whose implementation passes `python test_slugify.py` in that workspace.
2. Among passing candidates, choose the implementation with the fewest lines in `slugify.py`.
3. If there is still a tie, prefer the simplest implementation.

You must:

- copy the winning `slugify.py` to the run root
- write a final `RESULT.md` explaining:
  - which mapper won
  - which candidates passed or failed tests
  - each candidate's line count
  - why the winner was selected
