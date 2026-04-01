from PIL import Image, ImageDraw, ImageFont
import os

# Data
labels = ["capacity", "swizzle_xor", "swizzle_ldmatrix"]
memory = [86.24, 67.62, 79.42]
dram = [86.24, 61.07, 79.42]
l1 = [84.05, 70.02, 52.05]
l2 = [32.88, 44.36, 62.63]
compute = [33.63, 67.62, 29.40]
duration = [36.97, 69.37, 45.81]

# Image params
W, H = 1200, 700
margin = 80
plot_w = W - 2*margin
plot_h = H - 2*margin
bg_color = (255,255,255)
axis_color = (0,0,0)

im = Image.new('RGB', (W,H), bg_color)
d = ImageDraw.Draw(im)

# Fonts
try:
    font = ImageFont.truetype('DejaVuSans.ttf', 14)
    font_b = ImageFont.truetype('DejaVuSans-Bold.ttf', 16)
except Exception:
    font = ImageFont.load_default()
    font_b = font

# Axes
left = margin
top = margin
right = W - margin
bottom = H - margin

d.line((left, top, left, bottom), fill=axis_color, width=2)
d.line((left, bottom, right, bottom), fill=axis_color, width=2)

# Title (centered)
title = "GPU Speed Of Light — Throughput vs Duration"
try:
    tw, th = font_b.getsize(title)
except Exception:
    try:
        tw = d.textlength(title, font=font_b)
        th = 16
    except Exception:
        tw, th = (len(title) * 7, 16)
d.text(((W - tw) / 2, 20), title, fill=(0,0,0), font=font_b)

# Bars layout
n = len(labels)
# layout: divide group width and make bars adjacent (small fixed gap)
group_w = plot_w / n
total_bars = 5
# Reserve an inner portion of the group for the adjacent bars; leave outer spacing between groups
group_inner_frac = 0.7  # fraction of group width used for bars (0 < frac <= 1)
inner_w = group_w * group_inner_frac
gap = 0
# Make columns 25% narrower
reduced_width_factor = 0.75
bar_w = (inner_w / total_bars) * reduced_width_factor
# center the now-narrower set of bars within the group's inner region
inner_total = total_bars * bar_w + (total_bars - 1) * gap
group_left_offset = (group_w - inner_total) / 2
# positions: five bars centered in each group
colors = [(78,121,167),(242,142,43),(225,87,89),(118,183,178),(89,161,79)]
vals = [memory, dram, l1, l2, compute]
labels_bars = ['Memory Throughput (%)','DRAM Throughput (%)','L1/TEX Cache Throughput (%)','L2 Cache Throughput (%)','Compute (SM) Throughput (%)']

max_throughput = 110.0

for i in range(n):
    gx = left + i*group_w
    cx = gx + group_w/2
    # draw group label
    # PIL.ImageDraw.Text has different APIs across versions; use font.getsize if available
    try:
        text_w, text_h = font.getsize(labels[i])
    except Exception:
        text_w, text_h = d.textlength(labels[i], font=font), 10
    d.text((cx - text_w/2, bottom + 8), labels[i], fill=(0,0,0), font=font)
    # bars (adjacent inside the group's inner width)
    for j, arr in enumerate(vals):
        v = arr[i]
        # compute bar rect inside inner group area
        gx_left = gx + group_left_offset
        bx = gx_left + j * (bar_w + gap)
        # scale value to plot_h
        bh = (v / max_throughput) * (plot_h * 0.9)
        x0 = bx
        x1 = bx + bar_w
        y1 = bottom
        y0 = bottom - bh
        d.rectangle([x0,y0,x1,y1], fill=colors[j], outline=(0,0,0))

# Left y-axis labels
for k in range(0, 7):
    val = k * 20
    y = bottom - (val / max_throughput) * (plot_h * 0.9)
    d.line((left-6, y, left, y), fill=axis_color)
    txt = f"{val}%"
    d.text((left-60, y-8), txt, fill=(0,0,0), font=font)

# Plot duration on secondary axis at right
# Choose a tighter right-axis range so the duration line isn't stretched awkwardly
min_dur = min(duration)
max_dur = max(duration)
dur_margin = (max_dur - min_dur) * 0.3 if max_dur != min_dur else max_dur * 0.2
max_dur_plot = max_dur + dur_margin
# draw right axis
d.line((right, top, right, bottom), fill=axis_color, width=1)
for k in range(0,5):
    val = k * (max_dur/4)
    y = bottom - (val / max_dur) * (plot_h * 0.9)
    d.line((right, y, right+6, y), fill=axis_color)
    txt = f"{int(val)}ms"
    d.text((right+10, y-8), txt, fill=(0,0,0), font=font)

# Draw duration as three centered dots per group with the numeric value above
for i in range(n):
    gx = left + i*group_w
    cx = gx + group_w/2
    # map duration to y
    y = bottom - (duration[i] / max_dur) * (plot_h * 0.9)
    # single centered dot for duration
    dot_radius = 5
    d.ellipse((cx-dot_radius, y-dot_radius, cx+dot_radius, y+dot_radius), fill=(0,0,0))
    # numeric annotation centered above the dot
    txt = f"{duration[i]:.1f} ms"
    try:
        tw, th = font.getsize(txt)
    except Exception:
        tw, th = (len(txt)*6, 10)
    d.text((cx - tw/2, y - th - 8), txt, fill=(0,0,0), font=font)

# Legend: upper-right inside plot area, left of the duration right-axis; no background box
rect_w = 320
entry_h = 18
padding = 8
# place legend left of the right axis
rect_x2 = right - 12
rect_x1 = max(left + 10, rect_x2 - rect_w)
rect_y1 = top + 6
ly = rect_y1 + padding
for j, name in enumerate(labels_bars):
    y = ly + j * (entry_h + 6)
    # color swatch
    d.rectangle((rect_x1 + padding, y, rect_x1 + padding + 18, y + 12), fill=colors[j])
    d.text((rect_x1 + padding + 24, y - 2), name, fill=(0,0,0), font=font)
# duration legend entry (line symbol)
dy = ly + len(labels_bars) * (entry_h + 6)
d_x = rect_x1 + padding + 12
dot_r = 4
d.ellipse((d_x-dot_r, dy+6-dot_r, d_x+dot_r, dy+6+dot_r), fill=(0,0,0))
d.text((rect_x1 + padding + 36, dy - 4), 'Duration (ms)', fill=(0,0,0), font=font)

# Save
out_dir = 'prof/images/run2'
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'throughput_chart.png')
im.save(out_path)
print('Wrote', out_path)
