import freetype
import numpy as np

from dataclasses import dataclass
from argparse import ArgumentParser
from PIL import Image

SAMPLE_WIDTH = 12

CHAR_LOOKUP = [' ', '.', ':', 'a', '@']

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--image-path", required=True)
    return parser.parse_args()

@dataclass
class CachedGlyph:
    bitmap: np.ndarray
    advance: float

class GlyphRenderer:
    def __init__(self):
        self.glyph_cache = {}
        self.face = freetype.Face("Hack-Regular.ttf")
        self.face.set_char_size(48 * 64)
    
    def render_char(self, char):
        cached = self.glyph_cache.get(char, None)
        if cached is None:
            self.face.load_char(char)
            bitmap = self.face.glyph.bitmap

            if bitmap.pixel_mode != freetype.FT_PIXEL_MODE_GRAY:
                raise RuntimeError("Unsupported pixel mode")

            if bitmap.num_grays != 256:
                raise RuntimeError("Unsupported num_grays")
            
            cached = CachedGlyph(
                bitmap=np.array(bitmap.buffer, dtype=np.uint8).reshape((bitmap.rows, bitmap.width)),
                advance=self.face.glyph.advance.x / 64.0)
            
            self.glyph_cache[char] = cached
        
        return cached
    
    def row_height(self):
        return freetype.FT_MulFix(self.face.units_per_EM, self.face.size.y_scale) / 64.0

    def char_aspect(self):
        return self.render_char('@').advance / self.row_height()

def main(image_path):
    renderer = GlyphRenderer()
    SAMPLE_HEIGHT = int(SAMPLE_WIDTH / renderer.char_aspect())

    img = Image.open(image_path).convert(mode='L')
    img = np.array(img)

    output_img = np.zeros((
        int(img.shape[0] / SAMPLE_HEIGHT * renderer.row_height()),
        int(img.shape[1] / SAMPLE_WIDTH * renderer.row_height() * renderer.char_aspect())
    ), dtype=np.uint8)
    cursor_y = 0
    cursor_x = 0

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

            rendered = renderer.render_char(CHAR_LOOKUP[char_idx])
            output_y_end = cursor_y + rendered.bitmap.shape[0]
            output_x_end = cursor_x + rendered.bitmap.shape[1]
            try:
                output_img[cursor_y : output_y_end, cursor_x : output_x_end] = rendered.bitmap
            except:
                pass
            cursor_x += int(rendered.advance)

        cursor_x = 0
        cursor_y += int(renderer.row_height())
        
    Image.fromarray(output_img).save("test.png")

if __name__ == "__main__":
    main(**vars(parse_args()))