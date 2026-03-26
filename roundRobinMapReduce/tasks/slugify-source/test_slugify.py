from slugify import slugify


def check(input_text: str, expected: str) -> None:
    actual = slugify(input_text)
    assert actual == expected, f"{input_text!r} -> {actual!r}, expected {expected!r}"


check("Hello World", "hello-world")
check("  A B  C  ", "a-b-c")
check("already-good", "already-good")
check("Hello, World!", "hello-world")
check("Multiple---Hyphens", "multiple-hyphens")
check("snake_case + spaces", "snakecase-spaces")

print("ok")
