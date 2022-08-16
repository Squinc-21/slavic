from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
try:
    from font_utils import Font, try_glyphs
except ModuleNotFoundError:
    from avip.font_utils import Font, try_glyphs
try:
    from thresholding import balanced_thresholding
except ModuleNotFoundError:
    from avip.thresholding import balanced_thresholding

# Русские буквы написаны как есть, латинские и кириллические нерусские буквы написаны через их имена в Unicode
slavic_alphabet = [
    {'name': 'азъ', 'capital': 'А', 'small': 'а'},
    {'name': 'буки', 'capital': 'Б', 'small': 'б'},
    {'name': 'веди', 'capital': 'В', 'small': 'в'},
    {'name': 'глаголь', 'capital': 'Г', 'small': 'г'},
    {'name': 'добро', 'capital': 'Д', 'small': 'д'},
    {'name': 'есть', 'capital': '\N{cyrillic capital letter ie}',
     'small': '\N{cyrillic small letter ie}'},
    {'name': 'живите', 'capital': 'Ж', 'small': 'ж'},
    {'name': 'сало', 'capital': '\N{cyrillic capital letter dze}', 'small': '\N{cyrillic small letter dze}'},
    {'name': 'земля', 'capital': 'З', 'small': 'з'},
    {'name': 'иже', 'capital': 'И', 'small': 'и'},
    {'name': 'и', 'capital': '\N{cyrillic capital letter byelorussian-ukrainian i}',
     'small': '\N{cyrillic small letter byelorussian-ukrainian i}'},
    {'name': 'како', 'capital': 'К', 'small': 'к'},
    {'name': 'люди', 'capital': 'Л', 'small': 'л'},
    {'name': 'мыслите', 'capital': 'М', 'small': 'м'},
    {'name': 'нашъ', 'capital': 'Н', 'small': 'н'},
    {'name': 'онъ',
     'capital': [
         '\N{cyrillic capital letter round omega}',
         '\N{latin capital letter O}'
     ],
     'small': [
         '\N{cyrillic small letter round omega}',
         '\N{latin small letter O}'
     ]
     },
    {'name': 'омега',
     'capital': [
         '\N{cyrillic capital letter omega}',
         '\N{latin capital letter W}'
     ],
     'small': [
         '\N{cyrillic small letter omega}',
         '\N{latin small letter W}'
     ],
     },
    {'name': 'покой', 'capital': 'П', 'small': 'п'},
    {'name': 'рцы', 'capital': 'Р', 'small': 'р'},
    {'name': 'слово', 'capital': 'С', 'small': 'с'},
    {'name': 'твердо', 'capital': 'Т', 'small': 'т'},
    {'name': 'укъ',
     'capital': [
         '\N{cyrillic capital letter uk}',
         '\N{latin capital letter U}'
     ],
     'small': [
         '\N{cyrillic small letter uk}',
         '\N{latin small letter U}'
     ],
     },
    {'name': 'ферть', 'capital': 'Ф', 'small': 'ф'},
    {'name': 'херъ', 'capital': 'Х', 'small': 'х'},
    {'name': 'оть',
     'capital': [
         '\N{cyrillic capital letter ot}',
         '\N{latin capital letter T}'
     ],
     'small': [
         '\N{cyrillic small letter ot}',
         '\N{latin small letter T}'
     ]
     },
    {'name': 'цы', 'capital': 'Ц', 'small': 'ц'},
    {'name': 'червь', 'capital': 'Ч', 'small': 'ч'},
    {'name': 'ша', 'capital': 'Ш', 'small': 'ш'},
    {'name': 'ща', 'capital': 'Щ', 'small': 'щ'},
    {'name': 'еръ', 'capital': 'Ъ', 'small': 'ъ'},
    {'name': 'еры', 'capital': 'Ы', 'small': 'ы'},
    {'name': 'ерь', 'capital': 'Ь', 'small': 'ь'},
    {'name': 'ять',
     'capital': [
         '\N{cyrillic capital letter yat}',
         'Э'
     ],
     'small': [
         '\N{cyrillic small letter yat}',
         'э'
     ]
     },
    {'name': 'ю', 'capital': 'Ю', 'small': 'ю'},
    {'name': 'йотированный юсъ малый',
     'capital': [
         '\N{cyrillic capital letter iotified little yus}',
         '\N{latin capital letter K}'
     ],
     'small': [
         '\N{cyrillic small letter iotified little yus}',
         '\N{latin small letter K}'
     ]
     },
    {'name': 'юсъ малый',
     'capital': [
         '\N{cyrillic capital letter little yus}',
         '\N{latin capital letter Z}'
     ],
     'small': [
         '\N{cyrillic small letter little yus}',
         '\N{latin small letter Z}'
     ]
     },
    {'name': 'кси',
     'capital': [
         '\N{cyrillic capital letter ksi}',
         '\N{latin capital letter X}'
     ],
     'small': [
         '\N{cyrillic small letter ksi}',
         '\N{latin small letter X}'
     ]
     },
    {'name': 'пси',
     'capital': [
         '\N{cyrillic capital letter psi}',
         '\N{latin capital letter P}'
     ],
     'small': [
         '\N{cyrillic small letter psi}',
         '\N{latin small letter P}'
     ]
     },
    {'name': 'фита', 'capital': '\N{cyrillic capital letter fita}', 'small': '\N{cyrillic small letter fita}'},
    {'name': 'ижица',
     'capital': [
         '\N{cyrillic capital letter izhitsa}',
         '\N{latin capital letter V}'
     ],
     'small': [
         '\N{cyrillic small letter izhitsa}',
         '\N{latin small letter V}'
     ]
     },
]

slavic_font_names = [
    'AkathUcs8.ttf',
    # 'bukvica.ttf',
    'evngucs.ttf',
    'FedorovskUnicode.otf',
    'feofanucs.ttf',
    'Flavius.ttf',
    # 'HirmUcs8.ttf',
    # 'IndUcs.ttf',
    # 'IrmUcs.ttf',
    # 'KathUcs8.ttf',
    'MenaionUnicode.otf',
    # 'OglUcs8.ttf',
    # 'OstgUcs8.ttf',
    # 'PochUcs.ttf',
    # 'PosUcs8.ttf',
    # 'PsalUcs.ttf',
    # 'SlavUcs.ttf',
    # 'StusUcs.ttf',
    'TriUcs.ttf',
    # 'VertUcs.ttf',
    'Vilnius-Regular.otf',
    # 'ZlatUcs.ttf'
]


def crop_white(img: Image.Image) -> Image.Image:
    np_img = 255 - np.asarray(img)
    top, bottom = None, None
    left, right = None, None
    for i, value in enumerate(np_img.sum(axis=1)):
        if value > 0 and top is None:
            top = i
        if value > 0 and bottom is not None:
            bottom = None
        if value == 0 and bottom is None and top is not None:
            bottom = i
    if bottom is None:
        bottom = np_img.shape[0]
    for i, value in enumerate(np_img.sum(axis=0)):
        if value > 0 and left is None:
            left = i
        if value > 0 and right is not None:
            right = None
        if value == 0 and right is None and left is not None:
            right = i
    if right is None:
        right = np_img.shape[1]
    if left is None or top is None:
        raise Exception(f'{left}-{right}-{top}-{bottom}')
    return Image.fromarray(255 - np_img[top:bottom, left:right])


def letter_to_image(letter: str, font: Font, width: int, height: int):
    img = Image.new('L', (width, height), color=255)
    d = ImageDraw.Draw(img)
    w, h = d.textsize(letter, font=font.imageFont)
    point = (width - w) // 2, (height - h) // 2
    # print(w, h)
    d.text(point, letter, font=font.imageFont, stroke_fill=0, align="center")
    return img


def draw_letters(font_name, font_size=200):
    font = Font(Path(__file__).parent / 'fonts' / font_name, font_size)
    font_name = font_name.split('.')[0]
    for letter in slavic_alphabet:
        capital = try_glyphs(font, letter['capital'])
        directory = Path() / "avip/slavic_letters" / letter['name']
        directory.mkdir(exist_ok=True)
        if capital is not None:
            image = letter_to_image(capital, font=font, height=400, width=400)
            image = balanced_thresholding(image)
            image = crop_white(image)
            image.save(f"avip/slavic_letters/{letter['name']}/capital.{font_name}.{font_size}.jpg")
        small = try_glyphs(font, letter['small'])
        if small is not None:
            image = letter_to_image(small, font=font, height=400, width=400)
            image = balanced_thresholding(image)
            image = crop_white(image)
            image.save(f"avip/slavic_letters/{letter['name']}/small.{font_name}.{font_size}.jpg")


def remove_small_duplicates():
    """If small letter is the same as capital letter, small letter should be removed"""
    for directory in list(Path(__file__).parent.joinpath('slavic_letters').iterdir()):
        if not directory.is_dir():
            continue
        filenames = list(directory.iterdir())
        capital_filenames = sorted(filter(lambda f: f.name.startswith('capital'), filenames),
                                   key=lambda f: f.name[::-1])
        small_filenames = sorted(filter(lambda f: f.name.startswith('small'), filenames), key=lambda f: f.name[::-1])

        i = 0
        for capital_filename in capital_filenames:
            if len(small_filenames) <= i:
                break
            small_filename = small_filenames[i]
            if capital_filename.name.split('.')[-2] != small_filename.name.split('.')[-2]:
                continue
            small_image = np.asarray(Image.open(str(small_filename)))
            capital_image = np.asarray(Image.open(str(capital_filename)))
            if small_image.shape[0] == capital_image.shape[0] \
                    and small_image.shape[1] == capital_image.shape[1] \
                    and (small_image - capital_image).sum() == 0:
                small_filename.unlink()
            i += 1


def remove_bad_images():
    bad_image = np.asarray(Image.open('avip/slavic_letters/bad-image.bmp'))
    for directory in list(Path(__file__).parent.joinpath('slavic_letters').iterdir()):
        if not directory.is_dir():
            continue
        for file in directory.iterdir():
            image = np.asarray(Image.open(str(file)))
            if bad_image.shape[0] == image.shape[0] \
                    and bad_image.shape[1] == image.shape[1] \
                    and (bad_image - image).sum() == 0:
                print(file)
                file.unlink()
