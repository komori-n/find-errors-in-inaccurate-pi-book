import os
import cv2
import pyocr
import csv
import glob
import numpy as np
from joblib import Parallel, delayed
import tqdm
from image_process import TextImage
from PIL import Image

tools = pyocr.get_available_tools()
if len(tools) == 0:
    raise ValueError("pyocr tool not found")

tool = tools[0]


def cleanup_digits(digits_like: str) -> str:
    return (
        digits_like.replace(".", "").replace("\n", "").replace("-", "").replace(" ", "")
    )


def sharpen_image(img: cv2.Mat) -> cv2.Mat:
    tmp_img = cv2.filter2D(img, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
    return cv2.bitwise_not(cv2.convertScaleAbs(cv2.bitwise_not(tmp_img), None, 1.6))


def read_cell_text(cell: cv2.Mat):
    naive_text = tool.image_to_string(
        Image.fromarray(cell), lang="eng", builder=pyocr.builders.DigitBuilder()
    )
    text = cleanup_digits(naive_text)
    if len(text) == 10:
        return text

    # Apply sharpen filter and try recognition again
    sharpen_text = tool.image_to_string(
        Image.fromarray(sharpen_image(cell)),
        lang="eng",
        builder=pyocr.builders.DigitBuilder(),
    )
    return cleanup_digits(sharpen_text)


def main(filepath: str):
    csv_filename = "nyan/" + os.path.basename(os.path.splitext(filepath)[0]) + ".csv"
    if os.path.isfile(csv_filename):
        # skip processing if the csv file already exists
        return

    print(f"{csv_filename}")
    text_image = TextImage(filepath)
    results = []
    for i in tqdm.tqdm(range(100)):
        row_text = []
        for j in range(11):
            try:
                cell = text_image.cell(i, j)
                text = read_cell_text(cell)

                if text != "":
                    # split the text by 10 characters
                    while len(text) > 15:
                        row_text.append(text[:10])
                        text = text[10:]
                    row_text.append(text)

            except IndexError:
                pass

        if len(row_text) > 0 and len(row_text[0]) < 9 and row_text[0].endswith("00"):
            row_text = row_text[1:]

        results.append(row_text)

    with open(csv_filename, "w") as f:
        writer = csv.writer(f)
        writer.writerows(results)


if __name__ == "__main__":
    Parallel(n_jobs=-1)(delayed(main)(path) for path in glob.glob("img/002.jpg"))
