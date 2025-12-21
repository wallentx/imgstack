import os
import sys
import sysconfig

try:
    import site
except Exception:
    site = None


def _add_path(p: str) -> None:
    if not p:
        return
    if not os.path.isdir(p):
        return
    if p not in sys.path:
        sys.path.append(p)


paths = []

env_extra = os.environ.get("EXTRA_SITE_PACKAGES") or os.environ.get("IMGSTACK_EXTRA_SITE_PACKAGES")
if env_extra:
    paths.extend(env_extra.split(os.pathsep))

if site:
    try:
        paths.extend(site.getsitepackages())
    except Exception:
        pass
    try:
        paths.append(site.getusersitepackages())
    except Exception:
        pass

try:
    cfg = sysconfig.get_paths()
    for key in ("platlib", "purelib"):
        val = cfg.get(key)
        if val:
            paths.append(val)
except Exception:
    pass

ver = f"{sys.version_info.major}.{sys.version_info.minor}"
paths.append(f"/data/data/com.termux/files/usr/lib/python{ver}/site-packages")

for candidate in paths:
    _add_path(candidate)
