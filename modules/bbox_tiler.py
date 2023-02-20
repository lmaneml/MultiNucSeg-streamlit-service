import random

import cv2
import numpy as np
from albumentations import (
                            Compose,
                            LongestMaxSize,
                            PadIfNeeded,
                            BboxParams
                            )


class BboxTiler:
    '''Class for transformation of bboxes into mosaic images
    bboxes (np.array): array with bboxes coordinates, Order: x0, y0, x1, y1 (Pascal format) 
    image (np.array): source image of bboxes
    matrix_size (list): shape of the matrix. Default: [8, 8]
    nucleus_size (int): size of the nucleus image on the matrix. Default: 32
    preprocess: preprocesses from albumentations. 
                If None, default is LongestMaxSize, PadIfNeeded. 
                Should include bbox_params argument.
    '''
    def __init__(self, 
                 bboxes: np.array, 
                 image: np.array, 
                 nucleus_size: int = 32, 
                 matrix_size: list = [8,8],
                 preprocess=None):
        
        self.bboxes = bboxes
        self.image = image

        self.bboxes_idx = [x for x in range(0, len(bboxes))]
        random.shuffle(self.bboxes_idx)

        self.image_shape = self.image.shape[:2]
        self.nucleus_size = nucleus_size
        self.matrix_size = matrix_size
        self.n_cells = self.matrix_size[0] * self.matrix_size[1]

        self.bboxes_split = []
        for i in range(0, len(bboxes), self.n_cells):
            self.bboxes_split.append(self.bboxes_idx[i:i + self.n_cells])

        if preprocess:
            self.preproc = preprocess 
        else:
            self.preproc = Compose([
                                    LongestMaxSize(max_size=self.nucleus_size-4, #add black border
                                                   interpolation=1, 
                                                   always_apply=True, 
                                                   p=1.),
                                    PadIfNeeded(min_height=self.nucleus_size , 
                                                min_width=self.nucleus_size , 
                                                border_mode=cv2.BORDER_CONSTANT, 
                                                value=[0,0,0], 
                                                p=1.)
                                    ], bbox_params=BboxParams(format='pascal_voc'))

    def __len__(self):
        return len(self.bboxes_split)

    def load_mosaic(self, bboxes: np.array, 
                          matrix_size: list = [8, 8], 
                          nucleus_size: int = 32):
        '''Collect nuclei images into mosaic format
        Args:
            bboxes (np.array): array bboxes coordinates, 
                               Indices: 0 - x0, 1 - y0, 2 - x1, 3 - y1
            matrix_size (list): shape of the matrix. Default: [8, 8]
            nucleus_size (int): size of the nucleus image. Default: 32
        Returns:
            img_mosaic (np.array): image with shape matrix_size * nucleus_size
            transformed_bboxs (np.array): coordinates of the nuclei on the image
        '''
        transformed_bboxs = []
        img_mosaic = np.full((nucleus_size * matrix_size[0], 
                              nucleus_size * matrix_size[1], 3), 0, dtype=np.float32) 
        x1, y1 = 0, 0
        counter = 0
        for x_ind in range(matrix_size[1]):
            y1 = 0
            for y_ind in range(matrix_size[0]):
                if counter>(len(bboxes)-1):
                    transformed_bboxs.append([0. for _ in range(5)])
                    y1 = y1+nucleus_size
                    counter += 1
                    continue
                bbox = bboxes[counter]
                bbox = bbox[1:]
                x2 = x1+nucleus_size
                y2 = y1+nucleus_size
                img = self.image[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
                
                trn_bbox = [0.,0.,bbox[2]-bbox[0],bbox[3]-bbox[1],0]
                transform_nuc = self.preproc(image=img, bboxes=[trn_bbox])
                transform_bbox = list(transform_nuc['bboxes'][0])
                transform_nuc = transform_nuc['image']
                img_mosaic[y1:y2, x1:x2] = transform_nuc[:,:] 
                transform_bbox[0] += x1
                transform_bbox[1] += y1
                transform_bbox[2] += x1
                transform_bbox[3] += y1
                transformed_bboxs.append(transform_bbox)
                
                y1 = y2
                counter += 1
            x1 = x2
        return img_mosaic, transformed_bboxs    
    
    def mosaic_split(self):
        '''Function to split the data into dataset'''
        for idxs in self.bboxes_split:
            bboxes = [self.bboxes[z] for z in idxs]
            for y in bboxes:
                y[0:5] = [int(x) for x in y[0:5]]
            matrix_img, coords = self.load_mosaic(bboxes, matrix_size=self.matrix_size, nucleus_size=self.nucleus_size)
            if len(bboxes) < self.n_cells:
                bboxes += [[0. for _ in range(6)] for x in range(self.n_cells-len(bboxes))]

            yield matrix_img, np.array(bboxes), np.array(coords)