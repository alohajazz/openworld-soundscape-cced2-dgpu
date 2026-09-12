#!/usr/bin/env python3
"""Compatibility entry point for the corrected manuscript DAPT run.

The earlier version of this file implemented the superseded post-encoder-mask
analysis and must not be used for the September 2026 manuscript. The audited
executed implementation is kept verbatim in
``scripts/dapt_train_beats_mam_fixed.py``.
"""

from scripts.dapt_train_beats_mam_fixed import main


if __name__ == "__main__":
    main()
