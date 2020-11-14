from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import pickle

def find_skeleton(arr, start_point, threshold=45):
    tmp_arr = arr.copy()
    queue = []
    res = []
    queue.append(start_point)
    res.append(start_point)
    head = start_point
    while len(queue) > 0:
        point = queue.pop()
        x, y = point
        # print(x, y)
        # print(tmp_arr[218, 737])
        tmp_arr[x, y, :] = 0
        for dx in [1, -1]:
            for dy in [1, -1]:
                # print(x, y)
                new_point = [min(x + dx, 899), y + dy]

                cont = False
                for r in res[:-1]:
                    if np.linalg.norm(np.array(r) - np.array(new_point)) < threshold * 0.8:
                        cont = True
                        break
                if cont:
                    continue

                # print(new_point, tmp_arr[new_point])
                if tmp_arr[new_point[0]][new_point[1]][0] == 255 and  tmp_arr[new_point[0]][new_point[1]][1] == 255 and \
                    tmp_arr[new_point[0]][new_point[1]][2] == 255:
                    
                    # print(new_point)
                    if np.linalg.norm(np.array(head) - np.array(new_point)) > threshold:
                        # print(new_point)
                        head = new_point
                        res.append(new_point)
                        for ele in queue:
                            tmp_arr[ele[0], ele[1], :] = 0
                        queue = []

                    queue.append(new_point)

    # print(len(res))
    # print(res)
    return res

def get_character_image(character, start_point, erosion, threshold, division):
    # sample text and font
    unicode_text = character
    font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSans.ttf", 1000, encoding="unic")

    # get the line size
    text_width, text_height = font.getsize(unicode_text)
    print(text_width, text_height)

    # create a blank canvas with extra space between lines
    canvas = Image.new('RGB', (text_width, text_height), "black")

    # draw the text onto the text canvas, and use black as the text color
    draw = ImageDraw.Draw(canvas)
    draw.text((0, 0), character, 'white', font)

    # save the blank canvas to a file
    # canvas.save("unicode-text.png", "PNG")
    # canvas.show()

    arr = np.asarray(canvas)
    arr = cv2.erode(arr, kernel=np.ones((erosion, erosion), dtype=np.uint8))
    cv2.imshow('arr', arr)
    cv2.waitKey()

    x = y = 0
    res = find_skeleton(arr, start_point, threshold=threshold)
    x = [p[0] / division for p in res]
    y = [p[1] / division for p in res]
    print(len(res))

    for p in res:
        arr[p[0]][p[1]][0] = 0
        arr[p[0]][p[1]][1] = 0
        arr[p[0]][p[1]][2] = 255
    cv2.imshow('chosen arr', arr)
    cv2.waitKey()
    
    # loc = {
    #         character: (x, y)
    #     }
    # with open('softgym/envs/rope_configuration.pkl', 'wb') as handle:
    #     pickle.dump(loc, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return x, y


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    characters = ['S', 'O', 'M', 'C', 'U']
    start_point = [[350, 550], [200, 390], [875, 120], [360, 600], [200, 125]]
    erosion = [55, 55, 58, 55, 60]
    threshold = [46, 46, 46, 42, 42]
    division = [46, 46, 46, 42, 42]

    characters = ['M']
    start_point = [[875, 120]]
    erosion = [58]
    threshold = [36]
    division = [36]

    locs = {}
    for idx in range(len(characters)):
        x, y = get_character_image(characters[idx], start_point[idx], erosion[idx], threshold[idx], division[idx])
        locs[characters[idx]] = [x, y]

    # with open('softgym/envs/rope_configuration-longer.pkl', 'wb') as handle:
    #     pickle.dump(locs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    