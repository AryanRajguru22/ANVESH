import sys, pymupdf
from PIL import Image
pdf, out, start, n, cols = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
d = pymupdf.open(pdf)
ims = []
for i in range(start, min(start + n, d.page_count)):
    pm = d[i].get_pixmap(dpi=52)
    ims.append(Image.frombytes("RGB", (pm.width, pm.height), pm.samples))
w, h = ims[0].size
rows = (len(ims) + cols - 1) // cols
sheet = Image.new("RGB", (w * cols, h * rows), "white")
for k, im in enumerate(ims):
    sheet.paste(im, ((k % cols) * w, (k // cols) * h))
sheet.save(out)
