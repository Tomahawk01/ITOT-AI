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
        assert n == 1

        y = torch.randint(0, self.img.shape[0] - self.sample_height, (n,))
        x = torch.randint(0, self.img.shape[1] - self.sample_width, (n,))
        y_for_comparison = int(y * float(self.img_for_comparison.shape[0]) / float(self.img.shape[0]))
        x_for_comparison = int(x * float(self.img_for_comparison.shape[1]) / float(self.img.shape[1]))

        data = self.img[y:y + self.sample_height, x:x + self.sample_width]
        data_for_comparison = self.img_for_comparison[
            y_for_comparison:int(y_for_comparison + self.glyph_renderer.row_height()),
            x_for_comparison:int(x_for_comparison + self.glyph_renderer.char_width())]

        best_idx = 0
        best_score = math.inf
        for idx, char in enumerate(CHAR_LOOKUP):
            item = self.glyph_renderer.render_char(char)
            item_tensor = torch.zeros((int(self.glyph_renderer.row_height()), int(self.glyph_renderer.char_width())))
            item_tensor[0:item.bitmap.shape[0], 0:item.bitmap.shape[1]] = torch.tensor(item.bitmap)

            score = (item_tensor - data_for_comparison).abs().sum()
            if score < best_score:
                best_score = score
                best_idx = idx

        Image.fromarray(data.numpy()).save("data.png")
        Image.fromarray(data_for_comparison.numpy()).save("data_for_comparison.png")
        print(best_idx, best_score, CHAR_LOOKUP[best_idx])
        return data, best_idx

def main(image_path):
    renderer = GlyphRenderer()
    sample_width = 12
    sample_height = int(sample_width / renderer.char_aspect())
    net = Network(sample_width * sample_height, len(CHAR_LOOKUP))
    sampler = ImgSampler(sample_width, sample_height, renderer, image_path)
    sampler.get_samples(1)

    return
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    i = 0

    while True:
        optimizer.zero_grad()
        inputs = torch.rand((4096, 1)) * math.pi * 2 - math.pi
        expected = torch.sin(inputs)
        output = net(inputs)
        loss = criterion(output, expected)
        loss.backward()
        optimizer.step()
        i += 1
        if i % 5000 == 0:
            print("loss", loss.item())
            net.eval()
            with torch.no_grad():
                with open("output.csv", "w") as f:
                    inputs = torch.arange(500) / 500.0 * math.pi * 2.0 - math.pi
                    expected = torch.sin(inputs)
                    outputs = net(inputs.reshape(-1, 1))

                    for i in range(len(inputs)):
                        f.write("{}, {}, {}\n".format(inputs[i], outputs[i][0], expected[i]))
            net.train()

if __name__ == "__main__":
    main(**vars(parse_args()))