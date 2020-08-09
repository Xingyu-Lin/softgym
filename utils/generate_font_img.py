from PIL import Image, ImageDraw, ImageFont
import numpy as np


def get_character_image(character):
    # sample text and font
    unicode_text = character
    font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSans.ttf", 1000, encoding="unic")

    # get the line size
    text_width, text_height = font.getsize(unicode_text)

    # create a blank canvas with extra space between lines
    canvas = Image.new('RGB', (text_width, text_height), "white")

    # draw the text onto the text canvas, and use black as the text color
    draw = ImageDraw.Draw(canvas)
    draw.text((0, -50), character, 'black', font)

    # save the blank canvas to a file
    # canvas.save("unicode-text.png", "PNG")
    # canvas.show()
    return np.mean(np.asarray(canvas), axis=-1)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    img = get_character_image('B')
    plt.imshow(img)
    plt.show()
