#!/usr/bin/env python3

import os
import sys
from pathlib import Path


def find_comments_in_file(filepath):
    """Find all comments in a Python file."""
    comments = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                stripped = line.strip()
                if stripped.startswith("#") and not stripped.startswith("#!"):
                    comments.append((line_num, line.rstrip()))
    except Exception as e:
        print(f"Error reading {filepath}: {e}", file=sys.stderr)
    return comments


def scan_directory(directory):
    """Recursively scan directory for Python files with comments."""
    directory = Path(directory)
    all_comments = {}

    for py_file in directory.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue

        comments = find_comments_in_file(py_file)
        if comments:
            all_comments[str(py_file)] = comments

    return all_comments


def main():
    if len(sys.argv) > 1:
        search_dir = sys.argv[1]
    else:
        search_dir = "src"

    if not os.path.exists(search_dir):
        print(f"Directory not found: {search_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning {search_dir} for comments...\n")

    results = scan_directory(search_dir)

    if not results:
        print("✓ No comments found! Code is self-documenting.")
        sys.exit(0)

    print(f"✗ Found comments in {len(results)} file(s):\n")

    total_comments = 0
    for filepath, comments in sorted(results.items()):
        print(f"{filepath}:")
        for line_num, line in comments:
            print(f"  Line {line_num}: {line}")
            total_comments += 1
        print()

    print(f"Total: {total_comments} comment(s) found")
    print("\n⚠ REMINDER: This project has a NO COMMENTS policy.")
    print("Refactor these comments into self-documenting code.")

    sys.exit(1)


if __name__ == "__main__":
    main()
