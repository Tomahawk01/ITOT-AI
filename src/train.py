from image_to_text import GlyphRenderer, CHAR_LOOKUP

from argparse import ArgumentParser
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--image-path", required=True)
    return parser.parse_args()

class Network(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.linear1 = nn.Linear(num_inputs, num_outputs)
    
    def forward(self, x):
        return self.linear1(x)

def get_data(w, h):
    pass

class ImgSampler:
    def __init__(self, w, h, glyph_renderer, img_path):
        self.sample_width = w
        self.sample_height = h
        self.glyph_renderer = glyph_renderer
        img_pil = Image.open(img_path).convert(mode='L')
        self.img = torch.tensor(np.array(img_pil))

        img_for_comparison = img_pil.resize((
            int(self.img.shape[0] / h * self.glyph_renderer.row_height()),
            int(self.img.shape[1] / w * self.glyph_renderer.char_width())
        ))
        self.img_for_comparison = torch.tensor(np.array(img_for_comparison))

    def get_samples(self, n):
        x = torch.randint(0, self.img.shape[1] - self.sample_width, (n,))
        y = torch.randint(0, self.img.shape[0] - self.sample_height, (n,))

        y_for_comparison = y * float(self.img_for_comparison.shape[0]) / float(self.img.shape[0])
        x_for_comparison = x * float(self.img_for_comparison.shape[1]) / float(self.img.shape[1])
        x_for_comparison = x_for_comparison.to(torch.int)
        y_for_comparison = y_for_comparison.to(torch.int)

        def indices_from_start_len(starts, length):
            indices = torch.arange(0, length)
            indices = indices.repeat((starts.shape[0], 1))
            return indices + starts.reshape(-1, 1)

        xs = indices_from_start_len(x, self.sample_width)
        ys = indices_from_start_len(y, self.sample_height)

        data = self.img[ys][:, :, xs[1]]

        xs_for_comparison = indices_from_start_len(x_for_comparison.to(torch.int), int(self.glyph_renderer.char_width()))
        ys_for_comparison = indices_from_start_len(y_for_comparison.to(torch.int), int(self.glyph_renderer.row_height()))
        data_for_comparison = self.img_for_comparison[ys_for_comparison][:, :, xs_for_comparison[1]]

        char_height = int(self.glyph_renderer.row_height())
        char_width = int(self.glyph_renderer.char_width())

        item_tensor = torch.zeros((len(CHAR_LOOKUP), char_height, char_width))
        for idx, char in enumerate(CHAR_LOOKUP):
            item = self.glyph_renderer.render_char(char)
            item_tensor[idx, 0:item.bitmap.shape[0], 0:item.bitmap.shape[1]] = torch.tensor(item.bitmap)

        score = (item_tensor - data_for_comparison.repeat((len(CHAR_LOOKUP), 1, 1)).reshape((n, len(CHAR_LOOKUP), char_height, char_width)))
        labels = torch.max(score.sum(dim=(2, 3)), dim=1)[1]

        return data, labels

def render_full_image(img, renderer, sample_width, sample_height, iter, net):
    output_img = np.zeros((
        int(img.shape[0] / sample_height * renderer.row_height()),
        int(img.shape[1] / sample_width * renderer.row_height() * renderer.char_aspect())
    ), dtype=np.uint8)
    cursor_y = 0
    cursor_x = 0

    for y in range(int(img.shape[0] / sample_height)):
        for x in range(int(img.shape[1] / sample_width)):
            y_start = y * sample_height
            y_end = y_start + sample_height
            x_start = x * sample_width
            x_end = x_start + sample_width

            sample = img[y_start : y_end, x_start : x_end]
            output = net(sample.reshape((1, sample_width * sample_height)).to(torch.float32))

            char_idx = torch.max(output, 1)[1]

            rendered = renderer.render_char(CHAR_LOOKUP[char_idx])
            output_y_end = cursor_y + rendered.bitmap.shape[0]
            output_x_end = cursor_x + rendered.bitmap.shape[1]
            output_img[cursor_y : output_y_end, cursor_x : output_x_end] = rendered.bitmap
            cursor_x += int(rendered.advance)

        cursor_x = 0
        cursor_y += int(renderer.row_height())
    Image.fromarray(output_img).save("{}.png".format(iter))

def main(image_path):
    renderer = GlyphRenderer()
    sample_width = 12
    sample_height = int(sample_width / renderer.char_aspect())
    net = Network(sample_width * sample_height, len(CHAR_LOOKUP))
    sampler = ImgSampler(sample_width, sample_height, renderer, image_path)

    optimizer = optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    i = 0

    while True:
        optimizer.zero_grad()
        try:
            data, labels = sampler.get_samples(200)
        except:
            continue

        data = data.reshape(data.shape[0], data.shape[1] * data.shape[2])
        data = data.to(torch.float32)
        output = net(data)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        i += 1
        if i % 1000 == 1:
            print("loss:", loss)
            render_full_image(sampler.img, renderer, sample_width, sample_height, i, net)
        #if i % 5000 == 0:
        #    print("loss", loss.item())
        #    net.eval()
        #    with torch.no_grad():
        #        with open("output.csv", "w") as f:
        #            inputs = torch.arange(500) / 500.0 * math.pi * 2.0 - math.pi
        #            expected = torch.sin(inputs)
        #            outputs = net(inputs.reshape(-1, 1))
#
        #            for i in range(len(inputs)):
        #                f.write("{}, {}, {}\n".format(inputs[i], outputs[i][0], expected[i]))
        #    net.train()

if __name__ == "__main__":
    main(**vars(parse_args()))