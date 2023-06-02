from PIL import Image, ImageDraw

def draw_point(image, point, color, size):
    draw = ImageDraw.Draw(image)
    draw.ellipse([(point[0] - size / 2, point[1] - size / 2),
                  (point[0] + size / 2, point[1] + size / 2)],
                 fill=color)
    return image

def draw_line(image, p1, p2, color, width):
    draw = ImageDraw.Draw(image)
    # import pdb;pdb.set_trace()
    draw.line((p1, p2), fill=color, width=width)
    return image