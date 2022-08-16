from ast import excepthandler
import cv2
import os
import sys
import math
from matplotlib.pyplot import close
import numpy as np
from PIL import Image as im
from math import inf
from avip.features import (weight, density, relative_center, relative_axial_moments, axial_moments,
                      orientation_angle, center, horizontal_projection, vertical_projection)
from avip.main import draw_all_letters
from prettytable import PrettyTable
from PIL import Image
from pathlib import Path

def make_bw_etalons():
    path = 'superscript_etalons'
    bw_path = 'superscript_etalons_bw'
    exclude_dirs = ['empty']
    for sym_dir in list(set(os.listdir(path))-set(exclude_dirs)):
        dir_path = os.path.join(path, sym_dir)
        for sym_path in os.listdir(dir_path):
            full_sym_path = os.path.join(dir_path, sym_path)
            sym = cv2.imread(full_sym_path)
            gray = cv2.cvtColor(sym, cv2.COLOR_BGR2GRAY)
            (_, thresh) = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            new_dir_path = os.path.join(bw_path, sym_dir)
            try:
                os.makedirs(new_dir_path)
            except FileExistsError:
                pass
            # cv2.imwrite(os.path.join(new_dir_path, sym_path), thresh)

def get_scalar_features() -> PrettyTable:
    t = PrettyTable(['id', 'letter', 'case', 'font', 'size', 'width', 'height', 'weight', 'density',
                     'x_center', 'y_center',
                     'x_rel_center', 'y_rel_center',
                     'x_axial', 'y_axial',
                     'x_rel_axial', 'y_rel_axial',
                     'orientation_angle'],
                    align='l')
    t.align['letter'] = 'l'
    dir_path = 'superscript_etalons_bw'
    id = 1
    uni_size = 'uni_size'
    for directory in sorted(list(Path(__file__).parent.joinpath(dir_path).iterdir())):
        if not directory.is_dir():
            continue
        for file in sorted(list(directory.iterdir())):
            case = 'lower' # file.name.split('.')[0]
            font =  'original' # file.name.split('.')[1]
            image = np.asarray(Image.open(str(file))) / 255
            h, w = image.shape
            x_center, y_center = center(image)
            x_rel_center, y_rel_center = relative_center(image)
            x_axial, y_axial = axial_moments(image)
            x_rel_axial, y_rel_axial = relative_axial_moments(image)
            orientation = orientation_angle(image)
            t.add_row(
                [id, directory.name, case, font, uni_size, h, w, weight(image), density(image),
                 x_center, y_center,
                 x_rel_center, y_rel_center,
                 x_axial, y_axial,
                 x_rel_axial, y_rel_axial,
                 orientation])
            id += 1
    return t

def make_scalar_features_superscript():
    with open('scalar_features_superscript.csv', 'w') as f:
            f.write(get_scalar_features().get_csv_string())

make_scalar_features_superscript()

# draw_all_letters()
# get_scalar_features()
# with open('scalar_features.csv', 'w') as f:
#     f.write(get_scalar_features().get_csv_string())