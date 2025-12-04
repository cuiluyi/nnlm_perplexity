import re


def read_file(file_path, encoding="utf-8"):
    with open(file_path, "r", encoding=encoding) as f:
        return f.read()


def write_file(file_path, content: str, encoding="utf-8"):
    with open(file_path, "w", encoding=encoding) as f:
        f.write(content)
