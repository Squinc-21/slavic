from pathlib import Path

import numpy as np
from PIL import Image
from prettytable import PrettyTable
import matplotlib.pyplot as plt
try:
    from features import (weight, density, relative_center, relative_axial_moments, axial_moments,
                      orientation_angle, center, horizontal_projection, vertical_projection)
except ModuleNotFoundError:
    from avip.features import (weight, density, relative_center, relative_axial_moments, axial_moments,
                      orientation_angle, center, horizontal_projection, vertical_projection)
try:
    from slavic import (draw_letters, slavic_font_names, remove_small_duplicates, remove_bad_images)
except ModuleNotFoundError:
    from avip.slavic import (draw_letters, slavic_font_names, remove_small_duplicates, remove_bad_images)

def draw_all_letters():
    # выбери, какие шрифты нужны
    # в некоторых шрифтах отсутствуют некоторые буквы

    for font in slavic_font_names:
        for size in range(31, 41):
            draw_letters(font, font_size=size)
    remove_small_duplicates()
    remove_bad_images()


def get_scalar_features() -> PrettyTable:
    t = PrettyTable(['id', 'letter', 'case', 'font', 'size', 'width', 'height', 'weight', 'density',
                     'x_center', 'y_center',
                     'x_rel_center', 'y_rel_center',
                     'x_axial', 'y_axial',
                     'x_rel_axial', 'y_rel_axial',
                     'orientation_angle'],
                    align='l')
    t.align['letter'] = 'l'
    id = 1
    for directory in sorted(list(Path(__file__).parent.joinpath('slavic_letters').iterdir())):
        if not directory.is_dir():
            continue
        for file in sorted(list(directory.iterdir())):
            case = file.name.split('.')[0]
            font = file.name.split('.')[1]
            size = file.name.split('.')[2]
            image = np.asarray(Image.open(str(file))) / 255
            h, w = image.shape
            x_center, y_center = center(image)
            x_rel_center, y_rel_center = relative_center(image)
            x_axial, y_axial = axial_moments(image)
            x_rel_axial, y_rel_axial = relative_axial_moments(image)
            orientation = orientation_angle(image)
            t.add_row(
                [id, directory.name, case, font, size, h, w, weight(image), density(image),
                 x_center, y_center,
                 x_rel_center, y_rel_center,
                 x_axial, y_axial,
                 x_rel_axial, y_rel_axial,
                 orientation])
            id += 1
    return t


def draw_projection(x, y, path: str, show: bool = False):
    fig = plt.figure()
    axes = fig.add_subplot(1, 1, 1)
    axes.plot(x, y)
    axes.set_ylim(bottom=0)
    axes.set_xlim(left=0)
    if show:
        fig.show()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def draw_projections():
    Path('slavic_letter_projections').mkdir(exist_ok=True)
    for directory in sorted(Path(__file__).parent.joinpath('slavic_letters').iterdir())[:1]:
        if not directory.is_dir():
            continue
        for file in sorted(directory.iterdir()):
            letter = directory.name
            filename = '.'.join(file.name.split('.')[:-1])

            image = Image.open(str(file))
            # image.show()
            np_image = np.asarray(image) / 255

            projection = vertical_projection(np_image)
            projection = projection[::-1]
            indices = np.arange(0, len(projection))
            draw_projection(projection, indices, f'slavic_letter_projections/{letter}.{filename}.vertical.png',
                            show=True)

            projection = horizontal_projection(np_image)
            indices = np.arange(0, len(projection))
            draw_projection(indices, projection, f'slavic_letter_projections/{letter}.{filename}.horizontal.png',
                            show=True)


# if __name__ == '__main__':
# draw_all_letters()
with open('scalar_features.csv', 'w') as f:
    f.write(get_scalar_features().get_csv_string())

    # draw_projections()
