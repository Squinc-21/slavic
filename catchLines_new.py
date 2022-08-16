import cv2
import os
import sys
import math
import numpy as np
from PIL import Image
from math import inf


def get_x_profile(img):
    width = len(img[0])
    height = len(img)
    x_profiles = []

    for x in range(width):
        bright = 0
        for y in range(height):
            pix = img[y][x]
            if pix > 220:  # BLACK:
                bright += 1
        x_profiles.append(bright)

    return x_profiles

def get_y_profile(img):
    width = len(img[0])
    height = len(img)
    y_profiles = []

    for y in range(height):
        bright = 0
        for x in range(width):
            pix = img[y][x]
            if pix > 220:  # BLACK
                bright += 1
        y_profiles.append(bright)

    return y_profiles

def get_y_profile_percent(img):
    profile = get_y_profile(img)
    return sum(x > 0 for x in profile) / len(img)

def get_x_profile_percent(img):
    profile = get_x_profile(img)
    return sum(x > 0 for x in profile) / len(img[0])

def get_red_profile(img):
    width = len(img[0])
    height = len(img)
    y_profiles = []

    for y in range(height):
        bright = 0
        for x in range(width):
            pix = img[y][x]
            if set(pix) == set([0, 0, 255]):  # RED
                bright += 1
        y_profiles.append(bright)

    return y_profiles

def get_percent_of_black(img):
    width = len(img[0])
    height = len(img)
    c = np.count_nonzero(img > 220) # black
    return c / width * height

def count_line_av(line, r_idx):
    line_indeces = []
    if len(line):
        for x in zip(*np.where(line == 1)):
            real_idx = r_idx + x[0] - 10 if x[0] <= 10 else r_idx + x[0] - 10
            line_indeces.append(real_idx)
    top_line_av = None if not len(line_indeces) else sum(line_indeces)/len(line_indeces)
    return top_line_av

def get_concat_h_blank(im1, im2, color=(255, 255, 255), indent=3):
    dst = Image.new('RGB', (im1.width + indent + im2.width, max(im1.height, im2.height)), color)
    h_indent_left = 0 if im1.height >= im2.height else im2.height - im1.height
    h_indent_right = 0 if im1.height <= im2.height else im1.height - im2.height
    dst.paste(im1, (0, h_indent_left))
    dst.paste(im2, (im1.width + indent, h_indent_right))
    return dst

def get_concat_v_blank(im1, im2, color=(255, 255, 255), indent=5):
    dst = Image.new('RGB', (max(im1.width, im2.width), im1.height + im2.height + indent), color)
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height + indent))
    return dst

class ImageProcessor():

    def __init__(self, image_path, parts):
        self.src_path = image_path
        self.src = cv2.imread(image_path)
        self.width = len(self.src[0])
        self.height = len(self.src)
        self.parts = parts
        self.start_ch = 3

        self.left = None
        self.right = None
        self.left_crop = None
        self.right_crop = None
        self.chunks = []
        self.lines = []
        self.left_deltas = {
            "ll_delta_x": 0,
            "lr_delta_x": 0,
            "ll_delta_y": 0,
            "lu_delta_y": 0
        }

        self.right_deltas = {
            "rl_delta_x": 0,
            "rr_delta_x": 0,
            "rl_delta_y": 0,
            "ru_delta_y": 0
        }
        self.av_interval = 0
        self.num_marks = 0
        self.lines_array = []
        self.hash_lines = {}
        self.formed_strings = []

    def divide(self):
        self.left = self.src[:, :1600]
        self.right = self.src[:, 1600:]
        gray = cv2.cvtColor(self.src, cv2.COLOR_BGR2GRAY)
        (T, threshInv) = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)
        threshInv = cv2.medianBlur(threshInv, 3)
        cv2.imwrite("divide.jpg", threshInv)
        cv2.imwrite("left.jpg", self.left)
        cv2.imwrite("right.jpg", self.right)

    def crop_left(self):
        src_ = self.left[200:self.height - 250, 400: - 10]
        h = len(src_)
        self.left_deltas["ll_delta_x"] += 400
        self.left_deltas["lr_delta_x"] += 10
        self.left_deltas["ll_delta_y"] += 250
        self.left_deltas["lu_delta_y"] += 200
        gray = cv2.cvtColor(src_, cv2.COLOR_BGR2GRAY)

        (T, threshInv) = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)
        threshInv = cv2.medianBlur(threshInv, 3)
        x = get_x_profile(threshInv)
        cv2.imwrite("left_crop.jpg", threshInv)
        for i, val in enumerate(reversed(x)):
            if val > 5:
                break
        for j, val in enumerate(x):
            if val > 5:
                break
        y = get_y_profile(threshInv)
        for v, val in enumerate(reversed(y)):
            if val > 0:
                break
        v = v - 30 if v < h - 30 and v >= 30 else v
        text = threshInv[:-v, :] if v > 0 else threshInv
        text = text[:, j:-i] if i > 0 else text[:, j:]

        self.left_deltas["ll_delta_x"] += j
        self.left_deltas["lr_delta_x"] += i
        self.left_deltas["lu_delta_y"] += v
        cv2.imwrite("left_text_b.jpg", text)
        crop = self.left[
            self.left_deltas["ll_delta_y"]:-self.left_deltas["lu_delta_y"],
            self.left_deltas["ll_delta_x"]:-self.left_deltas["lr_delta_x"]
        ]
        cv2.imwrite("left_text.jpg", crop)
        self.left_crop = crop
        self.left_draw = self.left_crop.copy()

    def crop_right(self):
        h = len(self.src)
        w = len(self.src[0])
        src = self.src[200:h-250, 10:w-300]
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

        (T, threshInv) = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)
        threshInv = cv2.medianBlur(threshInv, 3)
        x = get_x_profile(threshInv)
        cv2.imwrite("right_crop.jpg", threshInv)

    def make_chunks(self, side):
        src = self.left_crop.copy() if side == 'left' \
            else self.right_crop.copy()
        # src = cv2.imread('t1.png')
        w = len(src[0])
        height = len(src)
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        (T, binary) = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)
        step = math.floor(len(src[0]) / self.parts)
        binary =  cv2.medianBlur(binary, 3)
        cv2.imwrite('bin.jpg', binary)
        profiles = []
        pos = 0
        while pos < w - step:
            lines = [0 for i in range(height)]
            chunk = binary[:, pos: pos + step]
            y = get_y_profile(chunk)
            profiles.append(y)
            c = 0
            while pos > step and c < height:
                if y[c] == 0:
                    c += 1
                else:
                    if c > 0:
                        lines[c - 1] = 1
                    else:
                        lines[c] = 1

                    cv2.line(
                        src,
                        (pos, c),
                        (pos + step, c),
                        (255, 0, 0),
                        thickness=1)
                    while c < height and y[c] > 0:
                        c += 1
            self.lines.append(lines)
            self.chunks.append(chunk)
            # cv2.imwrite(f"left_chunks/chunk_{pos}.jpg", chunk)
            pos += step
        lines = [0 for i in range(height)]
        chunk = binary[:, pos:]
        y = get_y_profile(chunk)
        profiles.append(y)
        c = 0
        while c < height:
            if y[c] == 0:
                c += 1
            else:
                if c > 0:
                    lines[c - 1] = 1
                else:
                    lines[c] = 1
                cv2.line(
                    src,
                    (pos, c),
                    (pos + step, c),
                    (255, 0, 0),
                    thickness=1)

                while c < height and y[c] > 0:
                    c += 1
        self.lines.append(lines)
        self.chunks.append(chunk)
        cv2.imwrite("111itog.jpg", src)
        # img = Image.fromarray(np.full((1,1),0, dtype=np.uint8))
        # for profile in profiles:
        #     pr_img_ar = np.full((height, step), 0, dtype=np.uint8)
        #     for idx, el in enumerate(profile):
        #         pr_img_ar[idx][:el] = 255
        #     pr_img = Image.fromarray(pr_img_ar)
        #     img = get_concat_h_blank(img, pr_img, color=0, indent=0)
        # img.save('profiles.jpg')


    def check_for_one(self, ch_num, sl, i, h):
        if 1 in sl and i + 8 < h:
            sll = self.lines[ch_num][i + 1:i + 8]
            ch = self.chunks[ch_num][i + 1:i + 8]
            sum_ = ch.sum() / 255
            step = math.floor(len(self.left_crop[0]) / self.parts)
            if (ver := sum_ < 10):
                self.lines[ch_num][i] = 0
                cv2.line(
                    self.left_crop,
                    (ch_num * step, i),
                    (ch_num * step + step, i),
                    (0, 255, 0),
                    thickness=1)
            self.check_for_one(ch_num, sll, i + 7, h)
            if not ver:
                self.lines[ch_num][i + 1:i + 8] = [0, 0, 0, 0, 0, 0, 0]

    def remove_duplicates(self, ch_num, cur, h):
        step = math.floor(len(self.left_crop[0]) / self.parts)
        start = cur + 1 if cur < h - 1 else h - 1 
        end = cur + 8 if cur < h - 8 else h - 1
        sl = self.lines[ch_num][start:end]
        draw = False
        try:
            if 1 in sl:
                idx = sl.index(1)
                ch = self.chunks[ch_num][start:start+idx]
                sum_ = ch.sum() / 255
                self.remove_duplicates(ch_num, idx, h)
                if sum_ < 8:
                    self.lines[ch_num][cur] = 0
                    if draw:
                        cv2.line(
                            self.left_crop,
                            (ch_num * step, cur),
                            (ch_num * step + step, cur),
                            (0, 0, 255),
                            thickness=1)
                else:
                    self.lines[ch_num][idx] = 0
                    if draw:
                        cv2.line(
                            self.left_crop,
                            (ch_num * step, idx),
                            (ch_num * step + step, idx),
                            (0, 255, 0),
                            thickness=1)
            else:
                s = self.chunks[ch_num][start:end].sum() / 255
                if s <= 10:
                    if draw:
                        cv2.line(
                            self.left_crop,
                            (ch_num * step, cur),
                            (ch_num * step + step, cur),
                            (255, 255, 0),
                            thickness=1)
                    self.lines[ch_num][cur] = 0
                return
        except (ValueError, Exception):
            return

    def repair_lines(self, side):
        h = len(self.lines[0])
        for ch_num, _ in enumerate(self.lines):
            i = 0
            while i < h:
                if self.lines[ch_num][i] == 1:
                    # sl = lines[i + 1:i + 8]
                    # cv2.line(src, (ch_num*step, i), (ch_num*step+step, i), (0, 0, 255), thickness=1)
                    # self.check_for_one(ch_num, sl, i, h)
                    self.remove_duplicates(ch_num, i, h)
                i += 1
        # cv2.imwrite("ttt.jpg", s)

    def repair_subscript(self, side):
        src = self.left_crop.copy() if side == 'left' \
            else self.right_crop
        h = len(self.lines[0])
        step = math.floor(len(src[0]) / self.parts)
        for ch_num, lines in enumerate(self.lines):
            i = 0
            while i < h - 20:
                if lines[i] == 1:
                    sl = lines[i + 1:i + 20]
                    cv2.line(
                        src,
                        (ch_num * step, i),
                        (ch_num * step + step, i),
                        (255, 0, 0),
                        thickness=1)
                    while 1 in sl:
                        index = sl.index(1)
                        self.lines[ch_num][i + 1 + index] = 0
                        sl[index] = 0
                i += 1
        cv2.imwrite("test_lines_1.jpg", src)


    def get_av_interval(self):
        h = len(self.lines[0])
        for ch_num, lines in enumerate(self.lines):
            if ch_num > 3:#self.parts - 3:
                break
            i = 0
            while i < h:
                if lines[i] == 1:
                    c = i + 1
                    while c < h and lines[c] != 1:
                        c += 1
                    if c < h and lines[c] == 1:
                        print(c-i)
                        self.av_interval = (self.av_interval * self.num_marks + (c - i)) / (self.num_marks + 1)
                        self.num_marks += 1
                i += 1
        print(self.av_interval, self.num_marks)

    def init_lines_array(self):
        lines_init = self.lines[2]

    def append_new_mark(self, ch_num, height):
        self.hash_lines[f'{ch_num}_{height}'] = len(self.lines_array)
        return height

    def init_line_dict(self, height, ch_num):
        return {
            'av': height,
            'items': [0 if part != ch_num else self.append_new_mark(ch_num, height) for part in range(self.parts+1)]
        }

    def init_lines_ar(self, side):
        lines_init = self.lines[self.start_ch]
        _ = [self.lines_array.append(self.init_line_dict(h, self.start_ch)) for h, x in enumerate(lines_init) if x == 1]

    def get_mark_index(self, ar, h, upper=True):
        try:
            return h - ar.index(1) if upper else h + ar.index(1) + 1
        except ValueError:
            return None

    def append_to_line(self, cur_line, ch_num, h, new_mark=False):
        count_non_zero = sum(x > 0 for x in self.lines_array[cur_line]['items'])
        self.lines_array[cur_line]['av'] = (self.lines_array[cur_line]['av']*count_non_zero + h) / (count_non_zero + 1)
        self.lines_array[cur_line]['items'][ch_num] = h
        self.hash_lines[f'{ch_num}_{h}'] = cur_line
        if new_mark:
            self.lines[ch_num][h] = 1

    def fill_in_lines_array(self, side):
        self.init_lines_ar(side)
        for ch_num, marks_ar in enumerate(self.lines[:-1]):
            if ch_num < self.start_ch:
                continue
            for h, mark in enumerate(marks_ar):
                if mark == 1:
                    current_line = self.hash_lines.get(f'{ch_num}_{h}', None)
                    if current_line == None:
                        continue
                    next_upper_index = self.get_mark_index(list(reversed(self.lines[ch_num+1][:h+1])), h)
                    next_lower_index = self.get_mark_index(self.lines[ch_num+1][h+1:], h, False)
                    current_line_av = self.lines_array[current_line]['av']
                    upper_line_av = self.lines_array[current_line - 1]['av'] if current_line > 0 else None
                    lower_line_av = self.lines_array[current_line + 1]['av'] if current_line < len(self.lines_array) - 1 else None

                    if next_upper_index == None:
                        upper_is_candidate = False
                    elif upper_line_av == None and next_lower_index != None and abs(next_upper_index-current_line_av) > abs(next_lower_index-current_line_av):
                        upper_is_candidate = False
                    elif upper_line_av != None and abs(next_upper_index-upper_line_av) < abs(next_upper_index-current_line_av):
                        upper_is_candidate = False
                    else:
                        upper_is_candidate = True

                    if next_lower_index == None or lower_line_av == None:
                        lower_is_candidate = False
                    elif lower_line_av == None and next_upper_index != None and abs(next_lower_index-current_line_av) > abs(next_upper_index-current_line_av):
                        lower_is_candidate = False
                    elif lower_line_av != None and abs(next_lower_index-lower_line_av) < abs(next_lower_index-current_line_av):
                        lower_is_candidate = False
                    else:
                        lower_is_candidate = True
                    
                    if upper_is_candidate: #and lower
                        self.append_to_line(current_line, ch_num+1, next_upper_index)
                    elif lower_is_candidate:
                        self.append_to_line(current_line, ch_num+1, next_lower_index)
                    else:
                        self.append_to_line(current_line, ch_num+1, h)
                        self.lines[ch_num+1][h] = 1

    def fix_lines_array(self):
        for line_num, line in enumerate(self.lines_array):
            for ch_num, height in enumerate(line['items'][:-1]):
                interval = self.lines[ch_num][self.lines_array[line_num-1]['items'][ch_num]+1:height] \
                    if line_num > 0 else self.lines[ch_num][0:height]
                interval = list(reversed(interval))
                try:
                    idx = interval.index(1)
                    slice_ = self.chunks[ch_num][height-1-idx:height]
                    y_profile_percent = get_y_profile_percent(slice_)
                    x_profile_percent = get_x_profile_percent(slice_)
                    if y_profile_percent >= 0.4 or x_profile_percent >= 0.4:
                        self.lines_array[line_num]['items'][ch_num] = height-1-idx
                        self.lines[ch_num][height-1-idx] = 1
                        self.lines[ch_num][height] = 0
                except ValueError:
                    pass
                    # todo - search for previously unrecognised marks
        self.draw_lines_array(self.lines_array)
                    

    def draw_lines_array(self, array, key='items'):
        src = self.left_draw
        color = (0, 0, 255) if key == 'items' else (255, 0, 0)
        step = math.floor(len(self.left_crop[0]) / self.parts)
        for line in array:
            # print("Line: \n", line, "\n")
            for ch_num, height in enumerate(line[key][:-1]):
                if height == 0:
                    continue
                if key == 'items':
                    # for i in range(ch_num * step, (ch_num + 1) * step, 7):
                    #     cv2.circle(src, (i, height-2), 2, color, -1)
                    #     # cv2.circle(src, (i, height, 2, color, -1)
                    cv2.line(
                            src,
                            (ch_num * step, height),
                            ((ch_num + 1) * step, height),
                            color,
                            thickness=3)
                else:
                    for i in range(ch_num * step, (ch_num + 1) * step, 7):
                        cv2.line(
                            src,
                            (i, height),
                            (i+2, height),
                            color,
                            thickness=3)
                if ch_num < self.parts - 1:
                    if key == 'items':
                        # begin = height if height < line[key][ch_num+1] else line[key][ch_num+1]
                        # end = line[key][ch_num+1] if height < line[key][ch_num+1] else height
                        # for i in range(begin, end, 6):
                        #     cv2.circle(src, ((ch_num + 1) * step, i), 2, color, -1)
                        cv2.line(
                            src,
                            ((ch_num + 1) * step, height),
                            ((ch_num + 1) * step, line[key][ch_num+1]),
                            color,
                            thickness=2)
                    else:
                        begin = height if height < line[key][ch_num+1] else line[key][ch_num+1]
                        end = line[key][ch_num+1] if height < line[key][ch_num+1] else height
                        for i in range(begin, end, 6):
                            cv2.line(
                            src,
                            ((ch_num + 1) * step, i),
                            ((ch_num + 1) * step, i+2),
                            color,
                            thickness=2)
        cv2.imwrite(f'draw_{key}.jpg', src)

    def add_lower_lines_old(self):
        for line_num, line in enumerate(self.lines_array):
            lower_line = []
            for ch_num, height in enumerate(line['items']):
                if ch_num < self.start_ch:
                    continue
                next_line = self.lines_array[line_num+1]['items'][ch_num] if line_num < len(self.lines_array) - 1 else len(self.chunks[0])

                slice_ = list(reversed(self.chunks[ch_num][height:next_line]))
                profile = get_y_profile(slice_)
                if not self.lines_array[line_num].get('lower', None):
                        self.lines_array[line_num]['lower'] = [0 for x in range(self.start_ch)]

                if profile[0] != 0:
                    self.lines_array[line_num]['lower'].append(next_line-1)
                else:
                    try:
                        lower_line = next(x for x, val in enumerate(profile) if val > 1)
                        self.lines_array[line_num]['lower'].append(next_line-lower_line-1)
                    except StopIteration:
                        if ch_num > 1:
                            lower_line = self.lines_array[line_num]['lower'][ch_num-1]
                            self.lines_array[line_num]['lower'].append(lower_line)
                        else:
                            self.lines_array[line_num]['lower'].append(next_line-1)
        self.draw_lines_array(self.lines_array, 'lower')

    def add_lower_lines(self):
        for line_num, line in enumerate(self.lines_array):
            lower_line = []
            for ch_num, height in enumerate(line['items']):
                if ch_num < self.start_ch:
                    continue
                next_line = None
                if line_num < len(self.lines_array) - 1:
                    for super in self.lines_array[line_num+1]['superscript']:
                        if super['ch_num'] == ch_num:
                            next_line = super['upper']
                    if not next_line:
                        next_line = self.lines_array[line_num+1]['items'][ch_num]
                else:
                    len(self.chunks[0])

                slice_ = self.chunks[ch_num][height+1:next_line]
                profile = get_y_profile(slice_)
                i = 0
                idx = None
                while i < len(profile) and profile[i] == 0:
                    i += 1
                while i < len(profile) - 2:
                    if profile[i] > 0  and profile[i+1] == 0 and profile[i+2] == 0:
                        idx = i
                        break
                    i += 1
                idx = next_line - 1 if idx is None else idx + height + 1
                profile_new = get_y_profile(self.chunks[ch_num][height+1:idx])
                if not len([x for x in profile_new if x > 2]):
                    idx = height + 1
                if not self.lines_array[line_num].get('lower', None):
                        self.lines_array[line_num]['lower'] = [0 for x in range(self.start_ch)]
                self.lines_array[line_num]['lower'].append(idx)
        self.draw_lines_array(self.lines_array, 'lower')


    def extract_superscript(self):
        for line_num, line in enumerate(self.lines_array):
            self.lines_array[line_num]['superscript'] = []
            ch_num = self.start_ch
            while ch_num < self.parts - 2:
                height = line['items'][ch_num]
                next_height = line['items'][ch_num+1]
                if next_height >= height:
                    ch_num += 1
                    continue
                next_slice = list(reversed(self.chunks[ch_num+1][next_height:height]))
                superscript = self.check_superscript(next_slice)
                set = False
                while superscript is not None and ch_num < self.parts - 2:
                    set = True
                    self.lines_array[line_num]['superscript'].append(
                        {
                            'ch_num': ch_num + 1,
                            'lower': height - superscript,
                            'upper': next_height
                        }
                    )
                    self.lines_array[line_num]['items'][ch_num+1] = height - superscript
                    ch_num += 1
                    next_height = line['items'][ch_num+1]
                    next_slice = list(reversed(self.chunks[ch_num+1][next_height:height]))
                    superscript = self.check_superscript(next_slice)
                if not set:
                    ch_num += 1
        # self.draw_lines_array(self.lines_array)
        self.print_superscript()

    def check_superscript(self, arr):
        if not arr:
            return None
        profile = get_y_profile(arr)
        i = 0
        while profile[i] > 0 and i < len(arr) - 1:
            i += 1
        profile = profile[i:]
        try:
            idx_zero = next(x for x, val in enumerate(profile) if val == 0)
            idx_non_zero = next(x for x, val in enumerate(profile) if val > 0)
            return idx_non_zero + i
        except StopIteration:
            return None

    def print_superscript(self):
        src = self.left_draw
        color = (0, 255, 0)
        step = math.floor(len(self.left_crop[0]) / self.parts)
        for line in self.lines_array:
            for idx, dict_ in enumerate(line['superscript']):
                cv2.line(
                        src,
                        (dict_['ch_num'] * step, dict_['lower']-1),
                        ((dict_['ch_num']+ 1) * step, dict_['lower']-1),
                        color,
                        thickness=2)
                cv2.line(
                        src,
                        (dict_['ch_num'] * step, dict_['upper']-1),
                        ((dict_['ch_num']+ 1) * step, dict_['upper']-1),
                        color,
                        thickness=2)
        cv2.imwrite(f'superscript.jpg', src)
        self.cut_superscript()


    def cut_superscript(self):
        step = math.floor(len(self.left_crop[0]) / self.parts)
        for l_num, line in enumerate(self.lines_array):
            n = 0
            i = 0
            l = len(line['superscript'])
            while i < l:
                cur_superscript = line['superscript'][i]
                low = cur_superscript['lower']
                up = cur_superscript['upper']
                left = cur_superscript['ch_num'] * step
                right = (cur_superscript['ch_num']+1) * step
                while i + 1 < l and line['superscript'][i+1]['ch_num'] - cur_superscript['ch_num'] == 1:
                    right = (line['superscript'][i+1]['ch_num']+1) * step
                    low = line['superscript'][i+1]['lower'] if line['superscript'][i+1]['lower'] > low else low
                    up = line['superscript'][i+1]['upper'] if line['superscript'][i+1]['upper'] < up else up
                    i += 1
                    cur_superscript = line['superscript'][i]
                im = self.left_crop[up:low+1, left:right]
                cv2.imwrite(f"superscript/{l_num}_{left-(step*self.start_ch)}_{right-(step*self.start_ch)}.jpg", im)
                n += 1                
                i += 1

    def cut_off_lines(self):
        step = math.floor(len(self.left_crop[0]) / self.parts)
        # etalon_len = len(self.left_crop[0]) - self.start_ch*step # arr -> self.parts+1
        etalon_len = self.parts*step - self.start_ch*step
        src = self.left_crop.copy()
        for l_num, line in enumerate(self.lines_array):
            # self.lines_array[l_num]['items'].append(line['items'][-1])
            # self.lines_array[l_num]['lower'].append(line['lower'][-1])
            highest = min(line['items'][self.start_ch:])
            lowest = max(line['lower'][self.start_ch:])
            arr = self.left_crop[highest:lowest+1, self.start_ch*step:(self.parts)*step]
            for ch_num, pair in enumerate(zip(line['items'][self.start_ch:], line['lower'][self.start_ch:]), 0):
                left_slice = arr[:,:ch_num*step]
                right_slice = arr[:, (ch_num+1)*step:]
                chunk = arr[:,ch_num*step:(ch_num+1)*step]
                upper, lower = pair
                chunk_slice = self.chunks[ch_num+self.start_ch][upper:lower]
                chunk_profile = get_y_profile(chunk_slice) if len(chunk_slice) else []
                inserted = False
                if len([x for x in chunk_profile if x > 2]):
                    chunk_inserted = chunk[upper-highest:len(chunk)-(lowest-lower)]
                    inserted = True
                else:
                    chunk_inserted = np.full((lower-upper+1, step, 3), 255)
                if ch_num < self.parts - self.start_ch or not inserted:
                    chunk_upper = np.full((upper-highest, step, 3), 255)
                else:
                    last_step = len(self.left_crop[0]) - self.parts*step
                    chunk_upper = np.full((upper-highest, last_step, 3), 255)
                if ch_num < self.parts - self.start_ch or not inserted:
                    chunk_lower = np.full((lowest-lower, step, 3), 255)
                else:
                    last_step = len(self.left_crop[0]) - self.parts*step
                    chunk_lower = np.full((lowest-lower, last_step, 3), 255)
                if not len(chunk_inserted):
                    chunk = np.concatenate((chunk_upper, chunk_lower), axis=0)
                elif len(chunk_inserted[0]):
                    chunk = np.concatenate((chunk_upper, chunk_inserted, chunk_lower), axis=0)
                arr = np.concatenate((left_slice, chunk, right_slice), axis=1)
            if len(arr[0]) > etalon_len:
                arr = arr[:, :etalon_len]

            cv2.imwrite(f'lines/{l_num}.jpg', arr)

    def cut_off_letters_rectangles(self):
        path = 'lines'
        num = 0
        for line_num, img in enumerate(sorted(os.listdir(path), key=lambda x:int(x.split('.')[0]))):
            image_path = os.path.join(path, img)
            src_ = cv2.imread(image_path)
            
            gray = cv2.cvtColor(src_, cv2.COLOR_BGR2GRAY)
            (T, threshInv) = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)
            (_, textArea) = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
            
            x_profile = get_x_profile(threshInv)
            h = len(src_)
            w = len(x_profile)
            i = 0
            rect_num = 0
            while i < w:
                if x_profile[i] == 0:
                    i += 1
                elif x_profile[i] > 0:
                    # cv2.line(src_, (i, 0), (i, h-1), (0,0,255))
                    left = i
                    while i < w and x_profile[i] > 0 or i < w - 1 and x_profile[i] == 0 and x_profile[i+1] > 0:
                        i += 1
                    right = i
                    stripe = textArea[:, left:right+1]
                    idx = 0
                    bottom = h
                    y_profile = get_y_profile(stripe)
                    while idx < bottom:
                        while idx < h and y_profile[idx] == 0:
                            idx += 1
                        top = idx
                        while idx < h and y_profile[idx] > 0:
                            idx += 1
                        bottom = idx
                    letter_rectangle = src_[top:bottom, left:right]
                    rectangle_path = f"letter_rectangles/{line_num}"
                    try:
                        os.makedirs(rectangle_path)
                    except FileExistsError:
                        pass
                    cv2.imwrite(rectangle_path+f'/{rect_num}_{left}_{right}.jpg', letter_rectangle)
                    rect_num += 1
                    # cv2.line(src_, (left, top), (left, bottom), (255,255, 0), 2)
                    # for i in range(top, bottom, 7):
                    #     cv2.line(src_, (right, i), (right, i+3), (255,255, 0), 2)

                    # cv2.line(src_, (right, top), (right, bottom), (255,255, 0))
                    # cv2.line(src_, (left, top), (right, top), (255,255, 0))
                    # cv2.line(src_, (left, bottom), (right, bottom), (255,255, 0))
                i += 1
            
            # cv2.imwrite(f"lines_bw/{num}.jpg", threshInv)
            # cv2.imwrite(f"lines_colored/{num}.jpg", src_)
            # cv2.imwrite(f"lines_colored1/{num}.jpg", src_)
            num += 1

    def cut_off_letters(self):
        rectangles_path = 'letter_rectangles'
        letters_base_path = 'letters'
        line_pathes = [os.path.join(rectangles_path, line) for line in os.listdir(rectangles_path)]
        thresh = 135
        for line_num, line_path in enumerate(sorted(line_pathes, key=lambda x:int(x.split('/')[-1]))):
            rect_pathes = [os.path.join(line_path, rect) for rect in os.listdir(line_path)]
            for rect_num, rect_path in enumerate(sorted(rect_pathes, key=lambda x:int(x.split('/')[-1].split('.')[0]))):
                rect = cv2.imread(rect_path)
                rect_gray = cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY)
                # b_t = balanced_thresholding(rect_gray)
                # (T, threshInv) = cv2.threshold(rect_gray, 140, 255, cv2.THRESH_BINARY_INV)
                # b,g,r = cv2.split(rect)
                _, threshInv = cv2.threshold(rect_gray, thresh, 255, cv2.THRESH_BINARY_INV)
                y_profile = get_y_profile(threshInv)
                i = 0
                l = len(y_profile)
                while i < l:
                    while i < l and y_profile[i] == 0:
                        i += 1
                    top = i
                    while i < l and y_profile[i] > 0:
                        i += 1
                    bottom = i
                    break
                # print(top, bottom)
                if top >= bottom:
                    continue
                letter = threshInv[top:bottom]

                letter_dir = os.path.join(letters_base_path, f"{line_num}")
                try:
                    os.makedirs(letter_dir)
                except FileExistsError:
                    pass
                rect = rect_path.split('/')[-1]
                letter_path = os.path.join(letter_dir, rect)
                cv2.imwrite(letter_path, 255-letter)

    def superscript_to_bw(self):
        superscript_path = 'superscript'
        superscript_bw_path = 'superscript_bw'
        thresh = 145
        for sup_sym_p in os.listdir(superscript_path):
            sup_sym_path = os.path.join(superscript_path, sup_sym_p)
            print(sup_sym_path)
            sup_sym = cv2.imread(sup_sym_path)
            sup_gray = cv2.cvtColor(sup_sym, cv2.COLOR_BGR2GRAY)
            _, threshInv = cv2.threshold(sup_gray, thresh, 255, cv2.THRESH_BINARY_INV)
            x_profile = get_x_profile(threshInv)
            i = 0
            l = len(x_profile)
            split_ = sup_sym_p.split('_')
            left = int(split_[1])
            right = left
            y_profile_percent = 0
            while i < l:
                while i < l and x_profile[i] == 0:
                    i += 1
                cur_left = i
                while i < l and x_profile[i] > 0:
                    i += 1
                cur_right = i
                cur_y_profile_percent = get_y_profile_percent(threshInv[:, cur_left:cur_right])
                if cur_y_profile_percent > y_profile_percent:
                    right = cur_right
                    left = cur_left
                    y_profile_percent = cur_y_profile_percent
            if left >= right:
                continue
            superscript = threshInv[:, left:right]
            sup_sym_p_new = split_[0] + '_' + f"{int(split_[1])+left}" + '_' + f"{int(split_[1])+right}" + '.jpg'
            superscript_path_new = os.path.join(superscript_bw_path, sup_sym_p_new)
            cv2.imwrite(superscript_path_new, 255-superscript)

    def form_strings(self):
        base_path = 'letters'
        superscript_dir = 'superscript_bw'
        superscripts = {}
        for sup_p in sorted(os.listdir(superscript_dir), key=lambda x:int(x.split('_')[1])):
            split_ = [int(x) if i < 2 else int(x.split('.')[0]) for i,x in enumerate(sup_p.split('_'))]
            key = split_[0]
            sup_path = os.path.join(superscript_dir, sup_p)
            try:
                superscripts[key].append({'left':split_[1], 'right':split_[2], 'path':sup_path})
            except KeyError:
                superscripts[key] = [{'left':split_[1], 'right':split_[2], 'path':sup_path}]
        formed_strings = []
        for dir_path in sorted(os.listdir(base_path), key=lambda x:int(x)):
            const_path = os.path.join(base_path, dir_path)
            formed_string = []
            sym_pathes = [x for x in sorted(os.listdir(const_path), key=lambda x:int(x.split('_')[0]))]
            superscript_symbols = superscripts.get(int(dir_path), None)
            for path in sym_pathes:
                split_ = [int(x) if i < 2 else int(x.split('.')[0]) for i,x in enumerate(path.split('_'))]
                left = split_[1]
                # right = split_[2]
                if superscript_symbols and superscript_symbols[0]['left'] <= left:
                    formed_string.append(superscript_symbols[0]['path'])
                    superscript_symbols = superscript_symbols[1:]
                formed_string.append(os.path.join(const_path, path))
            formed_strings.append(formed_string)
        self.formed_strings = formed_strings

    def classify(self):
        from classification import load_features, calculate_distance
        from avip.features import get_local_features

        inline = 'letters'
        inline_images_path = 'avip/slavic_letters'
        superscript_images_path = 'superscript_etalons_bw'
        superscript = 'superscript_bw'
        classification_dir = 'strings_classified'
        # outfile = 'res.txt'
        etalons_inline = load_features('scalar_features.csv')
        etalons_superscript = load_features('scalar_features_superscript.csv')
        # res_file = open(outfile, "w")

        image_pathes_arr = []
        for str_num, string in enumerate(self.formed_strings):
            image_pathes = []
            # res_file.write(f"Строка: {str_num}\n")
            local_features = []
            for sym in string:
                image = np.asarray(Image.open(str(sym))) / 255
                local_features.append(get_local_features(image))
            # str_ = ""
            for i, sym_features in enumerate(local_features):
                is_inline = string[i].split('/')[0] == inline
                if is_inline:
                    res = calculate_distance(etalons_inline, sym_features)
                else:
                    res = calculate_distance(etalons_superscript, sym_features)
                # res_file.write(f"Символ {i}:\n")
                predictions = [None for x in range(len(res))]
                for idx, (symbol, v) in enumerate(res.items()):
                    dir_ = inline_images_path if is_inline else superscript_images_path
                    sym_path_ = os.path.join(dir_, symbol)
                    path_ = os.path.join(sym_path_,'small.Flavius.31.jpg') if is_inline else os.path.join(sym_path_, '1.jpg')
                    if not os.path.exists(path_):
                        path_ = os.path.join(sym_path_, os.listdir(sym_path_)[0])
                    predictions[idx] = path_
                image_pathes.append(predictions)
            image_pathes_arr.append(image_pathes)
        self.create_exit_image(image_pathes_arr)



if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt

    image_path = '1.jpg'
    parts = 50
    im_proc = ImageProcessor(image_path, parts)
    im_proc.imitation()
    im_proc.divide()
    im_proc.crop_left()
    im_proc.crop_right()
    im_proc.make_chunks('left')
    im_proc.repair_lines('left')
    im_proc.repair_subscript('left')
    # im_proc.get_av_interval()
    im_proc.fill_in_lines_array('left')
    im_proc.fix_lines_array()
    im_proc.extract_superscript()
    im_proc.add_lower_lines()
    im_proc.cut_off_lines()
    im_proc.cut_off_letters_rectangles()
    im_proc.cut_off_letters()
    im_proc.superscript_to_bw()
    im_proc.form_strings()
    im_proc.classify()
    # im_proc.construct_strokes('left')
    # # chunks = make_first_chunk(left_crop, chunks)
    # im_proc.right_left_repair('left')
