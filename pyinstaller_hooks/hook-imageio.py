"""Collect only the ImageIO plugin used by imgstack.

The upstream hook collects every optional ImageIO plugin. In a Termux system
site environment that pulls unrelated desktop modules such as GTK and
Matplotlib into PyInstaller's analysis.
"""

from PyInstaller.utils.hooks import collect_data_files, copy_metadata


datas = collect_data_files("imageio", subdir="resources") + copy_metadata("imageio")
hiddenimports = ["imageio.plugins.pillow", "imageio.plugins.pillow_legacy"]
