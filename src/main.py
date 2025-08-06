import numpy as np

from argparse import ArgumentParser
from PIL import Image

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--image-path", required=True)
    return parser.parse_args()

def main(image_path):
    img = Image.open(image_path).convert(mode='L')
    img = np.array(img)
    print(img.shape, img.dtype)

if __name__ == "__main__":
    main(**vars(parse_args()))