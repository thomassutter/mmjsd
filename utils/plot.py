import textwrap
import torch
from torchvision.utils import save_image
from torchvision.utils import make_grid
from torchvision import transforms
from utils import text as text

from PIL import ImageDraw
from PIL import ImageFont
from PIL import Image

FONT = ImageFont.truetype('/home/thomas/FreeSerif.ttf',
                          38)

def create_fig(fn, img_data, num_img_row, save_samples=False):
    if save_samples:
        save_image(img_data.data.cpu(), fn, nrow=num_img_row);
    grid = make_grid(img_data, nrow=num_img_row);
    plot = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy();
    return plot;


def text_to_pil(t, imgsize, alphabet):
    w = 128; h=128;
    blank_img = torch.ones([3, w, h]);
    pil_img = transforms.ToPILImage()(blank_img.cpu()).convert("RGB")
    draw = ImageDraw.Draw(pil_img)
    text_sample = text.tensor_to_text(alphabet, t)[0]
    lines = textwrap.wrap(''.join(text_sample), width=8)
    y_text = h
    num_lines = len(lines);
    for l, line in enumerate(lines):
        width, height = FONT.getsize(line)
        draw.text((0, (h/2) - (num_lines/2 - l)*height), line, (0, 0, 0), font=FONT)
        y_text += height
    text_pil = transforms.ToTensor()(pil_img.resize((imgsize[1], imgsize[2]),
                                                    Image.ANTIALIAS));
    return text_pil;


def text_to_pil_celeba(t, imgsize, alphabet):
    w = 256; h=256;
    blank_img = torch.ones([3, w, h]);
    pil_img = transforms.ToPILImage()(blank_img.cpu()).convert("RGB")
    draw = ImageDraw.Draw(pil_img)
    text_sample = text.tensor_to_text(alphabet, t)[0]
    text_sample = ''.join(text_sample).translate({ord('*'): None})
    lines = textwrap.wrap(text_sample, width=16)
    y_text = h
    num_lines = len(lines);
    for l, line in enumerate(lines):
        width, height = FONT.getsize(line)
        draw.text((0, (h/2) - (num_lines/2 - l)*height), line, (0, 0, 0), font=FONT)
        y_text += height
    text_pil = transforms.ToTensor()(pil_img.resize((imgsize[1], imgsize[2]),
                                                    Image.ANTIALIAS));
    return text_pil;
