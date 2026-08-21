"""Keep Termux-owned shared libraries outside the frozen application."""

import pathlib
import sys


def pre_find_module_path(api):
    del api  # The hook only adjusts binary dependency filtering.

    from PyInstaller import compat
    from PyInstaller.depend import dylib

    if not compat.is_termux or getattr(dylib, "_imgstack_termux_filter", False):
        return

    system_lib_dir = (pathlib.Path(sys.base_prefix) / "lib").resolve()
    original_include_library = dylib.include_library

    def include_library(libname):
        path = pathlib.Path(libname).resolve()
        try:
            path.relative_to(system_lib_dir)
        except ValueError:
            return original_include_library(libname)

        # The bootloader needs the matching Python library in the bundle.
        return path.name.startswith("libpython")

    dylib.include_library = include_library
    dylib._imgstack_termux_filter = True
