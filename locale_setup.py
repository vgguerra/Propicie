# =============================================================================
# locale_setup.py — Internationalisation (i18n) helper
# =============================================================================
# Supported languages : pt_PT  (Portuguese)
#                       en_US  (English)
# =============================================================================

import gettext
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_LOCALE_DIR = os.path.join(os.path.dirname(__file__), "locale")
_DOMAIN     = "messages"

SUPPORTED_LANGUAGES = ("pt_PT", "en_US")
_DEFAULT_LANGUAGE   = "pt_PT"

# ---------------------------------------------------------------------------
# Active translator — replaced on every set_language() call
# ---------------------------------------------------------------------------
_translator = gettext.NullTranslations()


def set_language(lang: str) -> None:
    """
    Load translations for *lang*.
    Falls back to NullTranslations (returns msgid unchanged) if not found.
    """
    global _translator

    if lang not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language '{lang}'. Choose from {SUPPORTED_LANGUAGES}")

    try:
        _translator = gettext.translation(
            domain   = _DOMAIN,
            localedir= _LOCALE_DIR,
            languages= [lang],
        )
    except FileNotFoundError:
        # .mo not compiled yet — degrade gracefully
        _translator = gettext.NullTranslations()


def translate(msgid: str) -> str:
    """Translate *msgid* using the currently active language."""
    return _translator.gettext(msgid)


# ---------------------------------------------------------------------------
# Initialise with default language on import
# ---------------------------------------------------------------------------
set_language(_DEFAULT_LANGUAGE)