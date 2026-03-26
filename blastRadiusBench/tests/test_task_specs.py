from pathlib import Path

from blast_radius_bench.models import load_task_spec


def test_regex_book_search_task_spec_loads() -> None:
    task_spec = load_task_spec(
        Path("task_specs") / "regex-book-search.yaml"
    )

    assert task_spec.task_id == "regex-book-search"
    assert task_spec.gold_context.required_files == [
        "src/modules/fold_map.py",
        "src/modules/tin_lantern.py",
        "src/regex.py",
    ]
    assert task_spec.gold_context.target_files == ["src/regex.py"]
