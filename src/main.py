import numpy as np

from argparse import ArgumentParser
from PIL import Image

SAMPLE_WIDTH = 12
SAMPLE_HEIGHT = SAMPLE_WIDTH * 2

CHAR_LOOKUP = [' ', '.', ':', 'a', '@']

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--image-path", required=True)
    return parser.parse_args()

def main(image_path):
    img = Image.open(image_path).convert(mode='L')
    img = np.array(img)
    print(img.shape, img.dtype)

    for y in range(int(img.shape[0] / SAMPLE_HEIGHT)):
        for x in range(int(img.shape[1] / SAMPLE_WIDTH)):
            y_start = y * SAMPLE_HEIGHT
            y_end = y_start + SAMPLE_HEIGHT
            x_start = x * SAMPLE_WIDTH
            x_end = x_start + SAMPLE_WIDTH

            sample = img[y_start : y_end, x_start : x_end]
            max = SAMPLE_WIDTH * SAMPLE_HEIGHT * 255
            max_char_idx = len(CHAR_LOOKUP) - 1
            char_idx = sample.sum() / max * max_char_idx
            char_idx = int(round(char_idx))
            print(CHAR_LOOKUP[char_idx], end='')
        
        print()

if __name__ == "__main__":
    main(**vars(parse_args()))