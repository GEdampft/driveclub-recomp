"""raw2png.py FILE.raw... : tshot captures (u32 width, u32 height, RGBA8) to PNG beside them, alpha shown on grey."""
import struct, sys
from PIL import Image
for path in sys.argv[1:]:
    data = open(path, 'rb').read()
    w, h = struct.unpack('<II', data[:8])
    img = Image.frombytes('RGBA', (w, h), data[8:8 + w * h * 4])
    bg = Image.new('RGBA', (w, h), (128, 128, 128, 255))
    Image.alpha_composite(bg, img).convert('RGB').save(path[:-4] + '.png')
    alpha = img.getchannel('A')
    print(path, w, h, 'alpha min/max', alpha.getextrema())
