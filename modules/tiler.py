import math 
from typing import Tuple, Sequence, Iterable

import cv2
import numpy as np


class ImageSlicer:
    tile_size: Tuple[int, int]
    tile_step: Tuple[int, int]

    """
    Helper class to slice image into tiles.
    Adapted from https://github.com/BloodAxe/pytorch-toolbelt/blob/develop/pytorch_toolbelt/inference/tiles.py
    """

    def __init__(self, image_shape: Tuple[int, int], tile_size, tile_step=0, image_margin=0):
        """
        :param image_shape: Shape of the source image (H, W)
        :param tile_size: Tile size. Scalar or tuple (H, W)
        :param tile_step: Step in pixels between tiles. Scalar or tuple (H, W)
        :param image_margin:
        """
        self.image_height = image_shape[0]
        self.image_width = image_shape[1]

        if isinstance(tile_size, (np.ndarray, Sequence)):
            if len(tile_size) != 2:
                raise ValueError(f"Tile size must have exactly 2 elements. Got: tile_size={tile_size}")
            self.tile_size = int(tile_size[0]), int(tile_size[1])
        else:
            self.tile_size = int(tile_size), int(tile_size)

        if isinstance(tile_step, (np.ndarray, Sequence)):
            if len(tile_step) != 2:
                raise ValueError(f"Tile size must have exactly 2 elements. Got: tile_step={tile_size}")
            self.tile_step = int(tile_step[0]), int(tile_step[1])
        else:
            self.tile_step = int(tile_step), int(tile_step)

        if self.tile_step[0] < 1 or self.tile_step[0] > self.tile_size[0]:
            raise ValueError()
        if self.tile_step[1] < 1 or self.tile_step[1] > self.tile_size[1]:
            raise ValueError()

        overlap = (self.tile_size[0] - self.tile_step[0], self.tile_size[1] - self.tile_step[1])

        self.margin_left = 0
        self.margin_right = 0
        self.margin_top = 0
        self.margin_bottom = 0

        if image_margin == 0:
            # In case margin is not set, we compute it manually

            nw = max(1, math.ceil((self.image_width - overlap[1]) / self.tile_step[1]))
            nh = max(1, math.ceil((self.image_height - overlap[0]) / self.tile_step[0]))

            extra_w = self.tile_step[1] * nw - (self.image_width - overlap[1])
            extra_h = self.tile_step[0] * nh - (self.image_height - overlap[0])

            self.margin_left = extra_w // 2
            self.margin_right = extra_w - self.margin_left
            self.margin_top = extra_h // 2
            self.margin_bottom = extra_h - self.margin_top

        else:
            if isinstance(image_margin, Sequence):
                margin_left, margin_right, margin_top, margin_bottom = image_margin
            else:
                margin_left = margin_right = margin_top = margin_bottom = image_margin

            self.margin_left = margin_left
            self.margin_right = margin_right
            self.margin_top = margin_top
            self.margin_bottom = margin_bottom

        crops = []
        bbox_crops = []

        for y in range(
            0, self.image_height + self.margin_top + self.margin_bottom - self.tile_size[0] + 1, self.tile_step[0]
        ):
            for x in range(
                0, self.image_width + self.margin_left + self.margin_right - self.tile_size[1] + 1, self.tile_step[1]
            ):
                crops.append((x, y, self.tile_size[1], self.tile_size[0]))
                bbox_crops.append((x - self.margin_left, y - self.margin_top, self.tile_size[1], self.tile_size[0]))

        self.crops = np.array(crops)
        self.bbox_crops = np.array(bbox_crops)

    def iter_split(
        self, image: np.ndarray, border_type=cv2.BORDER_CONSTANT, value=0
    ) -> Iterable[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        if (image.shape[0] != self.image_height) or (image.shape[1] != self.image_width):
            raise ValueError()

        orig_shape_len = len(image.shape)

        for coords, crop_coords in zip(self.crops, self.bbox_crops):
            x, y, tile_width, tile_height = crop_coords
            x1 = max(x, 0)
            y1 = max(y, 0)
            x2 = min(image.shape[1], x + tile_width)
            y2 = min(image.shape[0], y + tile_height)

            tile = image[y1:y2, x1:x2]  
            if x < 0 or y < 0 or (x + tile_width) > image.shape[1] or (y + tile_height) > image.shape[0]:
                tile = cv2.copyMakeBorder(
                    tile,
                    top=max(0, -y),
                    bottom=max(0, y + tile_height - image.shape[0]),
                    left=max(0, -x),
                    right=max(0, x + tile_width - image.shape[1]),
                    borderType=border_type,
                    value=value,
                )

                # This check recovers possible lack of last dummy dimension for single-channel images
                if len(tile.shape) != orig_shape_len:
                    tile = np.expand_dims(tile, axis=-1)

            yield tile, coords

