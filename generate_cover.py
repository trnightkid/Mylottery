#!/usr/bin/env python3
"""
Generate a cinematic movie-poster style cover image.
Uses numpy vectorized ops + PIL ImageDraw (no slow per-pixel loops).
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

# ---------- config ----------
W, H = 1080, 1350
FONT_PATH = '/tmp/NotoSansCJK.otf'

# ---------- background: dark blue gradient via numpy ----------
bg = np.zeros((H, W, 3), dtype=np.uint8)
y_coords, x_coords = np.ogrid[:H, :W]
cx, cy = W // 2, H // 2
dist = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
max_dist = np.sqrt(cx**2 + cy**2)
ratio = np.clip(dist / max_dist, 0, 1)

bg[:,:,0] = np.clip(5  + (25 - 5)  * (1 - ratio * 1.4), 0, 255).astype(np.uint8)  # R
bg[:,:,1] = np.clip(8  + (18 - 8)  * (1 - ratio * 1.4), 0, 255).astype(np.uint8)  # G
bg[:,:,2] = np.clip(24 + (75 - 24) * (1 - ratio * 1.4), 0, 255).astype(np.uint8)  # B

img = Image.fromarray(bg, 'RGB')
draw = ImageDraw.Draw(img)

# ---------- fonts ----------
title_font    = ImageFont.truetype(FONT_PATH, int(H * 0.024))
subtitle_font = ImageFont.truetype(FONT_PATH, int(H * 0.016))
body_font     = ImageFont.truetype(FONT_PATH, int(H * 0.013))
gold_font     = ImageFont.truetype(FONT_PATH, int(H * 0.0145))
banner_font   = ImageFont.truetype(FONT_PATH, int(H * 0.022))
footer_font   = ImageFont.truetype(FONT_PATH, int(H * 0.010))

def draw_ball(draw, cx, cy, radius, rgb_tuple):
    """Draw a lottery ball with shading on img (PIL ImageDraw)."""
    r, g, b = rgb_tuple
    # Main filled circle
    draw.ellipse([cx-radius, cy-radius, cx+radius, cy+radius],
                 fill=(r, g, b))
    # Darker edge for depth
    draw.ellipse([cx-radius, cy-radius, cx+radius, cy+radius],
                 outline=(max(0,r-60), max(0,g-40), max(0,b-40)), width=2)

def ball_number(draw, cx, cy, num, radius, is_blue=False):
    """Draw number on ball."""
    fnt_size = int(radius * 1.0) if not is_blue else int(radius * 0.9)
    fnt = ImageFont.truetype(FONT_PATH, fnt_size)
    s = str(num)
    bw = draw.textbbox((0, 0), s, font=fnt)
    tw, th = bw[2]-bw[0], bw[3]-bw[1]
    # White with dark outline
    draw.text((cx - tw//2 + 1, cy - th//2 + 1), s, fill=(0,0,0), font=fnt)
    draw.text((cx - tw//2, cy - th//2), s, fill='white', font=fnt)

# ---------- lottery balls ----------
red_positions = [
    (int(W*0.65), int(H*0.52)),
    (int(W*0.74), int(H*0.60)),
    (int(W*0.60), int(H*0.61)),
    (int(W*0.70), int(H*0.68)),
    (int(W*0.80), int(H*0.54)),
    (int(W*0.83), int(H*0.65)),
]
blue_positions = [
    (int(W*0.68), int(H*0.41)),
    (int(W*0.80), int(H*0.38)),
]
red_r = int(H * 0.038)
blue_r = int(H * 0.030)

import random
random.seed(77)
red_nums  = random.sample(range(1, 34), 6)
blue_nums = random.sample(range(1, 17), 2)

for (cx, cy), num in zip(red_positions, red_nums):
    draw_ball(draw, cx, cy, red_r, (220, 38, 38))
    ball_number(draw, cx, cy, num, red_r)
for (cx, cy), num in zip(blue_positions, blue_nums):
    draw_ball(draw, cx, cy, blue_r, (37, 99, 235))
    ball_number(draw, cx, cy, num, blue_r, is_blue=True)

# ---------- semi-transparent header overlay ----------
header = Image.new('RGBA', (W, int(H*0.21)), (10, 15, 46, 200))
img.paste(header, (0, 0), mask=header)

# Gold decorative line
for x in range(int(W*0.06), int(W*0.94)):
    px = int(H*0.795)
    ratio = (x - W*0.06) / (W*0.88)
    brt = int(180 + 30 * abs(2*ratio - 1))
    draw.point((x, px),   fill=(brt, 160, 20))
    draw.point((x, px+1), fill=(20, 58, 130))

# ---------- helper: draw centered text ----------
def center_text(draw, text, y, font, color):
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    draw.text((W//2 - tw//2, y), text, fill=color, font=font)

# ---------- title ----------
center_text(draw, '【宣战书：对抗神谕的代码】', int(H*0.915), title_font, (251, 191, 36))
center_text(draw, 'LOTTERY  GAN  |  庄先森PBU',    int(H*0.855), subtitle_font, (148, 163, 184))

# ---------- quote block ----------
quotes = [
    (H*0.780, "在这个巨大的、贪婪的人类欲望生成对抗网络中，"),
    (H*0.758, "我们并非玩家，我们仅仅是数据。"),
    (H*0.736, "一边是名为「一夜暴富」的原始算法，"),
    (H*0.714, "它将命运操控于股掌之间，"),
    (H*0.692, "将希望提炼成维持系统运转的燃料。"),
    (H*0.670, "这就是那个「彩票GAN」——"),
    (H*0.648, "一个以亿万个灵魂的熵增为代价的闭环模型。"),
]
for y_norm, text in quotes:
    center_text(draw, text, int(y_norm), body_font, (210, 220, 235))

# ---------- gold divider ----------
for x in range(int(W*0.10), int(W*0.90)):
    ratio = (x - W*0.10) / (W*0.80)
    brt = int(40 + 15 * abs(2*ratio - 1))
    draw.point((x, int(H*0.627)), fill=(brt, brt//3, 130))
    draw.point((x, int(H*0.629)), fill=(15, 40, 110))

# ---------- gold declaration ----------
center_text(draw, "我是庄先森PBU，我拒绝成为这个模型中沉默的噪点。",
            int(H*0.610), gold_font, (251, 191, 36))

# ---------- bottom section ----------
bottom = [
    (H*0.575, "然而，我依然在这里。"),
    (H*0.555, "在这场博弈中，结局并不重要——",),
    (H*0.535, "宣战本身即是反抗。", (251, 191, 36)),
    (H*0.515, "即使我输给了概率，",),
    (H*0.495, "我也绝不想输给那个被你写好的命运。",),
]
for item in bottom:
    y_norm = item[0]
    text = item[1]
    color = item[2] if len(item) > 2 else (220, 230, 245)
    bold = "反抗" in text
    fnt = gold_font if bold else body_font
    center_text(draw, text, int(y_norm), fnt, color)

# ---------- banner ----------
banner_y = int(H * 0.34)
banner_h = int(H * 0.09)
banner_img = Image.new('RGBA', (W, banner_h), (30, 58, 138, 200))
banner_mask = Image.new('L', (W, banner_h), 200)
img.paste(banner_img, (0, banner_y), mask=banner_mask)

# Banner border lines
for x in range(int(W*0.12), int(W*0.88)):
    for dy in range(3):
        py1 = banner_y + dy
        py2 = banner_y + banner_h - 1 - dy
        draw.point((x, py1), fill=(59, 130, 246))
        draw.point((x, py2), fill=(59, 130, 246))

center_text(draw, '代码启动。博弈开始。',
            banner_y + banner_h//2 - 12, banner_font, (255, 255, 255))

# ---------- footer ----------
center_text(draw, 'GitHub: trnightkid/Mylottery  |  深度学习 · 双色球 · 2200+期历史数据',
            int(H*0.948), footer_font, (100, 116, 139))

# ---------- save ----------
output_path = '/home/openclaw/.openclaw/workspace/output/lottery_gan_cover.png'
img.save(output_path, 'PNG')
print(f"✅ 封面图已保存: {output_path}  ({W}x{H} px)")
