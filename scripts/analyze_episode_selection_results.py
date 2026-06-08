#!/usr/bin/env python
"""Regenerate aggregate tables and plots from completed episode-selection runs."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from episode_selection.runner import main


if __name__ == "__main__":
    args = sys.argv[1:] + ["--only-analyze"]
    main(args)
