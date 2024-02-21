import numpy as np
from PIL import Image


def draw(data, file):
    # Define image dimensions
    image_width = len(data[-1]) * 4  # Each cube in the row occupies 4 pixels
    image_height = len(data) * 4  # Each cube in the column occupies 4 pixels

    # Create an image with white background
    image = Image.new("RGB", (image_width, image_height), color=(255, 255, 255))

    # Create a numpy array from the image to easily manipulate pixels
    pixels = np.array(image)

    # Generate image based on data
    for row_idx, row in enumerate(data):
        for col_idx, value in enumerate(row):
            # Determine color based on the value
            color = (255 * (1 - value), 255 * (1 - value), 255 * (1 - value))

            # Determine pixel coordinates for the cube
            start_x = col_idx * 4
            end_x = start_x + 3
            start_y = row_idx * 4
            end_y = start_y + 3

            # Set color for the cube
            pixels[start_y : end_y + 1, start_x : end_x + 1] = color

    # Create an image from the numpy array and save it
    result_image = Image.fromarray(pixels.astype(np.uint8))
    result_image.save(file)


def draw_dir(dir):
    with open(f"{dir}/att-result.txt", "r") as f:
        import json

        alldata = json.load(f)
        for block in range(32):
            for header in range(32):
                data = alldata[f"{block}-{header}"]

                draw(data, f"{dir}/att-{block}-{header}.png")
        pass


def main():
    dirs = [f"../candle-examples/tests/llama_7b_chat/test_{i}" for i in range(100)]
    print(dirs)


draw_dir("../candle-examples/tests/llama_7b_chat_1240/test_1")
