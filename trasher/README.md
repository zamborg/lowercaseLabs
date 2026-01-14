# trasher

`trash` is a small `rm`-like CLI that moves files/directories into the macOS Trash instead of deleting them.

## Install (dev / editable)

```bash
python3 -m pip install -e .
```

If your `trash` executable isn’t found after installing, make sure your Python “scripts” directory is on `PATH`
(for the python.org installer this is often `/Library/Frameworks/Python.framework/Versions/<ver>/bin`).

## Usage

```bash
trash some-file.txt
trash -r some-directory/
trash -rf build/ dist/
trash -i important.txt
trash --dry-run -rv *
trash ls
trash ls -l --sort modified
trash restore some-file.txt
trash empty
```

If `trash` is already taken on your system, use `trasher` (installed by this project) or adjust your `PATH` so this project’s `trash` resolves first.

## Notes

- By default, items are moved into the appropriate macOS Trash folder:
  - Same volume as your home: `~/.Trash`
  - Other volumes: `<volume>/.Trashes/<uid>/`
- When run via `sudo`, the tool uses the invoking user’s Trash (via `SUDO_UID`/`SUDO_USER`) when available.
- `trash restore ...` uses a persistent map at `~/.trasher/map.json` and only works for items trashed by this tool.
- If `sudo trash ...` can’t find the executable, use `sudo $(which trash) ...` or add the install location to `sudo`’s `secure_path`.
- This tool moves items to Trash for recovery, but Finder “Put Back” metadata is not guaranteed.
