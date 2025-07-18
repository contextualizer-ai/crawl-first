"""
YAML utilities for crawl-first.

Handles custom YAML dumping with proper string formatting.
"""

from typing import Any

import yaml


class NoRefsDumper(yaml.SafeDumper):
    """Custom YAML dumper that doesn't use references/anchors."""

    def ignore_aliases(self, data: Any) -> bool:
        return True


def str_presenter(dumper: yaml.SafeDumper, data: str) -> yaml.ScalarNode:
    """Custom YAML string presenter that quotes strings with special characters."""
    if (
        "\n" in data
        or data.startswith("%")
        or any(
            char in data
            for char in [
                ":",
                "{",
                "}",
                "[",
                "]",
                ",",
                "#",
                "&",
                "*",
                "!",
                "|",
                ">",
                "'",
                '"',
                "`",
            ]
        )
    ):
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style='"')
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


NoRefsDumper.add_representer(str, str_presenter)
