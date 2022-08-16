from pathlib import PurePath
from typing import Union, List, Optional

from PIL import ImageFont
from fontTools.ttLib import TTFont


class Font:
    def __init__(self, path: Union[str, PurePath], size: int):
        self.ttFont = TTFont(str(path))
        self.imageFont = ImageFont.truetype(str(path), size=size)


# https://stackoverflow.com/questions/47948821/python-imagefont-and-imagedraw-check-font-for-character-support
def has_glyph(font: Font, glyph: str):
    for table in font.ttFont['cmap'].tables:
        if ord(glyph) in table.cmap.keys():
            return True
    return False


def try_glyphs(font: Font, glyphs: Union[str, List[str]]) -> Optional[str]:
    if isinstance(glyphs, str):
        glyphs = [glyphs]
    for glyph in glyphs:
        if has_glyph(font, glyph):
            return glyph
    return None
