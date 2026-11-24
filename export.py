"""Render the loop to a video file (no window needed).

  .venv/bin/python export.py                 -> sharingan.mp4, 1920x1080, 30 fps
  .venv/bin/python export.py --png 26 out.png  -> a single frame at t=26s
"""
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import main

FPS = 30
CW, CH, FONT_PX = 12, 24, 20           # cell size and font size in pixels (big enough to survive feed compression)
main.COLS, main.ROWS = 160, 45         # 1920 x 1080
OUT_W, OUT_H = 1920, 1080


def glyph_atlas():
    font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", FONT_PX, index=1)  # bold
    ascent, descent = font.getmetrics()
    top = (CH - (ascent + descent)) // 2
    atlas = np.zeros((main.NR, CH, CW), np.float32)
    for i, ch in enumerate(main.RAMP):
        im = Image.new("L", (CW, CH), 0)
        ImageDraw.Draw(im).text((0, top), ch, font=font, fill=255)
        atlas[i] = np.asarray(im, np.float32) / 255.0
    return atlas


PALETTE = np.array([[(k >> 8 & 15) * 17, (k >> 4 & 15) * 17, (k & 15) * 17]
                    for k in range(4096)], np.float32)


def frame(scene, atlas, t):
    idx, key = scene.render(t)
    g = atlas[idx]                                    # rows, cols, CH, CW
    rgb = g[..., None] * PALETTE[key][:, :, None, None, :]
    rgb = rgb.transpose(0, 2, 1, 3, 4).reshape(main.ROWS * CH, main.COLS * CW, 3)
    return rgb.astype(np.uint8)


def run():
    scene, atlas = main.Scene(CW, CH), glyph_atlas()
    if len(sys.argv) == 4 and sys.argv[1] == "--png":
        Image.fromarray(frame(scene, atlas, float(sys.argv[2]))).save(sys.argv[3])
        return
    import imageio_ffmpeg
    out = sys.argv[1] if len(sys.argv) > 1 else "sharingan.mp4"
    w, h = main.COLS * CW, main.ROWS * CH
    ff = subprocess.Popen(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-",
         "-vf", f"pad={OUT_W}:{OUT_H}:(ow-iw)/2:(oh-ih)/2:black",
         "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", out], stdin=subprocess.PIPE)
    n = int(main.LOOP * FPS)
    for i in range(n):
        ff.stdin.write(frame(scene, atlas, i / FPS).tobytes())
        if i % 150 == 0:
            print(f"frame {i}/{n}", flush=True)
    ff.stdin.close()
    if ff.wait() != 0:
        sys.exit("ffmpeg failed")
    print("wrote", out)


if __name__ == "__main__":
    run()
