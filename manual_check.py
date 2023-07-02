import csv
import sys
import os
from typing import IO, Optional

import cv2
from image_process import TextImage

ANS_FILENAME = "pi-1m.txt"


def show_question(img: cv2.Mat, test: str, ans: str) -> Optional[str]:
    cv2.imshow("manual check", img)
    cv2.waitKey(1)

    ret = None
    while True:
        user_input = input("please type correct: ").strip()
        if len(user_input) != 10:
            continue

        if user_input == test:
            break
        else:
            print(f"ans:   {ans}")
            print(f"test:  {test}")
            print(f"input: {user_input}")
            confirmation = input("(y/N/?): ").strip()
            if confirmation == "y":
                ret = user_input
                break
            elif confirmation == "?":
                ret = "??????????"
                break

    cv2.destroyWindow("manual check")
    return ret


def check_all(out: IO, text_image: TextImage, s_ans: str, s_test: str, page: int):
    assert len(s_ans) == len(s_test)

    skip_until = 0
    for i in range(len(s_ans)):
        if i < skip_until:
            continue

        if s_ans[i] != s_test[i]:
            chunk_start = i // 10 * 10
            chunk_end = chunk_start + 10
            substr_ans = s_ans[chunk_start:chunk_end]
            substr_test = s_test[chunk_start:chunk_end]
            skip_until = chunk_end

            row = chunk_start // 100
            col = (chunk_start // 10) % 10

            try:
                manual_input = show_question(
                    text_image.cell(row, col), substr_test, substr_ans
                )
                if manual_input is not None:
                    out.write(f"{page} {row} {col} {manual_input}\n")
            except:
                out.write(f"{page} {row} {col} ??????????\n")


def main(datapath: str):
    filepath = os.path.join(os.path.dirname(__file__), ANS_FILENAME)

    ans = ""
    with open(filepath, "r") as f:
        ans = "".join(f.readlines()).strip()

    ans = ans[2:]  # skip "3."

    out = open("check_out.txt", "a")
    for i in range(47, 101):
        csvpath = os.path.join(datapath, str(i).zfill(3) + ".csv")
        imgpath = os.path.join("img", str(i).zfill(3) + ".jpg")
        text_image = TextImage(imgpath)

        x = ""
        with open(csvpath, "r") as f:
            reader = csv.reader(f)
            x = "".join(["".join(r) for r in reader])

        check_all(out, text_image, ans[(i - 1) * 10000 : i * 10000], x, i)

    out.close()


if __name__ == "__main__":
    argv = sys.argv
    data_path = "./data"
    if len(argv) > 1:
        data_path = argv[1]
        if not os.path.isdir(data_path):
            print(f"usage {argv[0]} datadir")

    main(data_path)
