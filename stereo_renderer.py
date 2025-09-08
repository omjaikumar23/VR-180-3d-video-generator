import numpy as np
from PIL import Image

def create_stereo_frame(color_img, depth_map, eye_separation=15):
    color_arr = np.array(color_img)
    depth_map = np.array(depth_map)
    h, w = depth_map.shape

    left_img = np.zeros_like(color_arr)
    right_img = np.zeros_like(color_arr)

    for y in range(h):
        for x in range(w):
            shift = int(depth_map[y, x] * eye_separation)
            if x + shift < w:
                left_img[y, x + shift] = color_arr[y, x]
            if x - shift >= 0:
                right_img[y, x - shift] = color_arr[y, x]

    left_img_pil = Image.fromarray(left_img)
    right_img_pil = Image.fromarray(right_img)
    return left_img_pil, right_img_pil