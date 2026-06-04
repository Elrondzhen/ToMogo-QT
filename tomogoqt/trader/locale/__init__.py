import gettext
from pathlib import Path


localedir: Path = Path(__file__).parent

translations: gettext.GNUTranslations | gettext.NullTranslations = gettext.translation("tomogoqt", localedir=localedir, fallback=True)

_ = translations.gettext
