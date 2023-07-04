import numpy as np
import cv2
from typing import Callable


class TextImage:
    def __init__(self, filepath: str):
        self.img = _make_flat(cv2.imread(filepath))
        self._vertical_imgs = None
        self._cell_imgs = None

    def row(self, i: int, j: int) -> cv2.Mat:
        if self._vertical_imgs is None:
            self._make_vertical_images()

        return self._vertical_imgs[i]

    def cell(self, i: int, j: int) -> cv2.Mat:
        if self._cell_imgs is None:
            self._make_cell_images()

        return self._cell_imgs[i][j]

    def _make_vertical_images(self):
        vertical_rects, vertical_contours = _find_vertical_contours(self.img)
        vertical_imgs = _make_contour_images(
            self.img, vertical_rects, vertical_contours, 10
        )
        vertical_imgs = [cv2.resize(img, None, fx=2.0, fy=2.0) for img in vertical_imgs]

        self._vertical_imgs = vertical_imgs

    def _make_cell_images(self):
        if self._vertical_imgs is None:
            self._make_vertical_images()

        cell_imgs = []
        for img in self._vertical_imgs:
            rects, contours = _find_cell_contours(img)
            row_images = _make_contour_images(img, rects, contours, 4)
            cell_imgs.append(row_images)

        self._cell_imgs = cell_imgs


Rect = tuple[int, int, int, int]


def make_padding_func(padding: int, shape: tuple[int, int]) -> Callable[[Rect], Rect]:
    def padding_func(rect: Rect) -> Rect:
        x, y, w, h = rect

        x = max(x - padding, 0)
        y = max(y - padding, 0)
        b = min(y + h + padding * 2, shape[0])
        r = min(x + w + padding * 2, shape[1])

        w = r - x
        h = b - y
        return x, y, w, h

    return padding_func


def _make_flat(img: cv2.Mat) -> cv2.Mat:
    tmp_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tmp_img = cv2.bitwise_not(tmp_img)
    tmp_img = cv2.dilate(src=tmp_img, kernel=np.ones((10, 10)), iterations=5)
    _, tmp_img = cv2.threshold(tmp_img, 100, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(tmp_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) < 1000000:
            continue

        rect = cv2.minAreaRect(cnt)
        angle = rect[2]

        orig_height, orig_width, _ = img.shape
        rect_width, rect_height = rect[1]
        if rect_width > rect_height:
            angle += 90.0

        height, width = max(orig_height, orig_width), min(orig_height, orig_width)
        rot_center = (height / 2, height / 2)
        matrix = cv2.getRotationMatrix2D(rot_center, angle, 1.0)

        return cv2.warpAffine(img, matrix, (width, height), borderValue=(255, 255, 255))

    raise RuntimeError("Text area is not found")


def _find_vertical_contours(
    img: cv2.Mat,
) -> tuple[list[Rect], list[np.ndarray]]:
    tmp_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tmp_img = cv2.bitwise_not(tmp_img)
    tmp_img = cv2.dilate(src=tmp_img, kernel=np.ones((1, 10)), iterations=5)
    _, tmp_img = cv2.threshold(tmp_img, 90, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(tmp_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # sort by y
    sorted_contours = sorted(contours, key=lambda cnt: cv2.boundingRect(cnt)[1])

    def detect_text_box(cnt) -> bool:
        _x, _y, w, h = cv2.boundingRect(cnt)
        return w > 1400 and h < 50

    ret_contours = list(map(cv2.convexHull, filter(detect_text_box, sorted_contours)))
    ret_rects = list(map(cv2.boundingRect, ret_contours))

    return ret_rects, ret_contours


def _find_cell_contours(
    img: cv2.Mat,
) -> tuple[list[Rect], list[np.ndarray]]:
    tmp_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tmp_img = cv2.bitwise_not(tmp_img)
    tmp_img = cv2.dilate(src=tmp_img, kernel=np.ones((5, 5)), iterations=6)
    _, tmp_img = cv2.threshold(tmp_img, 20, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(tmp_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # sort by x
    sorted_contours = sorted(contours, key=lambda cnt: cv2.boundingRect(cnt)[0])

    def detect_text_box(cnt) -> bool:
        _x, _y, w, h = cv2.boundingRect(cnt)
        return 80 < w < 1000 and 35 < h

    ret_contours = list(map(cv2.convexHull, filter(detect_text_box, sorted_contours)))
    rects = map(cv2.boundingRect, ret_contours)
    padded_rects = list(map(make_padding_func(10, img.shape), rects))

    return padded_rects, ret_contours


def _make_contour_images(
    img: cv2.Mat, rects: list[Rect], contours: list[np.ndarray], padding: int
) -> list[cv2.Mat]:
    images = []
    padding_func = make_padding_func(padding, img.shape)
    for rect, cnt in zip(rects, contours):
        x, y, w, h = padding_func(rect)
        vertical_img = img[y : y + h, x : x + w].copy()

        # Remove outside of contour hull
        pts = cnt - cnt.min(axis=0) + padding
        mask = np.zeros(vertical_img.shape[:2], np.uint8)
        cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
        vertical_img[mask != 255] = (255, 255, 255)

        images.append(vertical_img)

    return images
