"""
AdvPost watermark for AI-generated images.

Stamps the official AdvPost logo icon (no wordmark, on a soft white round badge, see
assets/advpost_watermark.png) onto a generated image, in place.

Configuration (all optional, set in .env):
  WATERMARK_ENABLED       1 (default) / 0   turn the watermark on or off
  WATERMARK_POSITION      bottom-right (default) | bottom-left | top-right | top-left
  WATERMARK_WIDTH_PCT     badge width as % of image width (default 10, allowed 3-30)
  WATERMARK_MARGIN_PCT    gap from the edges as % of image width (default 3)
  WATERMARK_OPACITY       0.15-1.0 multiplier for the badge (default 0.55, a faint mark)
  WATERMARK_LOGO_PATH     override the badge image

Applying a watermark never fails a generation: on any error it is logged and the
original image is left untouched.
"""

import logging
import os
import threading

from PIL import Image

logger = logging.getLogger(__name__)

_DEFAULT_LOGO = os.path.join(os.path.dirname(__file__), "assets", "advpost_watermark.png")
_POSITIONS = {"top-right", "top-left", "bottom-right", "bottom-left"}
_MIN_BADGE_PX = 56

_logo_lock = threading.Lock()
_logo_cache: dict = {}


def _env_float(name: str, default: float, lo: float, hi: float) -> float:
    try:
        value = float(os.getenv(name, "").strip() or default)
    except ValueError:
        return default
    return max(lo, min(hi, value))


def watermark_enabled() -> bool:
    return os.getenv("WATERMARK_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}


def _load_logo() -> Image.Image:
    path = os.getenv("WATERMARK_LOGO_PATH", "").strip() or _DEFAULT_LOGO
    mtime = os.path.getmtime(path)
    with _logo_lock:
        cached = _logo_cache.get(path)
        if cached and cached[0] == mtime:
            return cached[1]
        with Image.open(path) as img:
            logo = img.convert("RGBA")
        _logo_cache[path] = (mtime, logo)
        return logo


def apply_watermark(image_path: str) -> bool:
    """Stamp the AdvPost badge onto image_path in place. Returns True if applied."""
    if not watermark_enabled():
        return False
    tmp_path = f"{image_path}.wm.tmp"
    try:
        logo = _load_logo()
        position = os.getenv("WATERMARK_POSITION", "bottom-right").strip().lower()
        if position not in _POSITIONS:
            position = "bottom-right"
        width_pct = _env_float("WATERMARK_WIDTH_PCT", 10.0, 3.0, 30.0)
        margin_pct = _env_float("WATERMARK_MARGIN_PCT", 3.0, 0.0, 15.0)
        opacity = _env_float("WATERMARK_OPACITY", 0.55, 0.15, 1.0)

        with Image.open(image_path) as src:
            had_alpha = "A" in src.getbands()
            base = src.convert("RGBA")

        badge_w = max(_MIN_BADGE_PX, int(base.width * width_pct / 100))
        badge_h = max(1, round(logo.height * badge_w / logo.width))
        badge = logo.resize((badge_w, badge_h), Image.LANCZOS)
        if opacity < 1.0:
            alpha = badge.getchannel("A").point(lambda a: int(a * opacity))
            badge.putalpha(alpha)

        margin = int(base.width * margin_pct / 100)
        x = margin if position.endswith("left") else base.width - badge_w - margin
        y = margin if position.startswith("top") else base.height - badge_h - margin
        base.alpha_composite(badge, (max(0, x), max(0, y)))

        out = base if had_alpha else base.convert("RGB")
        out.save(tmp_path, format="PNG", optimize=True)
        os.replace(tmp_path, image_path)
        return True
    except Exception:
        logger.exception("Could not apply AdvPost watermark to %s", image_path)
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        return False
