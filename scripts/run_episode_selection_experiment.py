#!/usr/bin/env python
"""Run semantic and quality-aware episode selection experiments."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from episode_selection.runner import main


if __name__ == "__main__":
    main()
