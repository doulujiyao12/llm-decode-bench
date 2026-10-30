"""Opt-in server hook, inherited by spawned Python workers through PYTHONPATH.

If you already use sitecustomize, integrate install_from_env into it rather than
shadowing it. Nothing is installed unless LDB_EVENT_DIR is explicitly set.
"""
import os
import sys

if os.environ.get("LDB_EVENT_DIR"):
    try:
        from llm_decode_bench.sglang_hook import install_from_env
        install_from_env()
    except Exception as error:
        print(f"llm-decode-bench hook not installed: {type(error).__name__}", file=sys.stderr)
