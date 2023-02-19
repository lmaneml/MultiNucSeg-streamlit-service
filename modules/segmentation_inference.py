import os
import json

import cv2
import numpy as np
import streamlit as st
import onnxruntime as ort
from stqdm import stqdm
from albumentations import Normalize

from .bbox_tiler import BboxTiler


def load_metadata(path, cell_info=False):
    with open(path, 'r') as fd:
        d = json.load(fd)
        if cell_info:
            d = {int(k):v for k, v in d.items()}
    return d

CELLS_DICT = load_metadata('./configs/classes.json', cell_info=True)
NAMES_DICT = {x['idx']:x['name'] for x in CELLS_DICT.values()}
COLORS = {x['name']:x['color'] for x in CELLS_DICT.values()}


class ObjectSegmentation:
    def __init__(self, model_path: str):

        self.nucl_size_px = 32
        self.matrix_size = [4,4]
        self.nn_input_size = [128,128]
        self.model_path = model_path
        
        self.model = self.load_onnx_model(self.model_path, cuda=False)

        self.output_result = []

    def tile_image_bbox(self, image, bboxes):

        tiler = BboxTiler(bboxes, 
                          image, 
                          nucleus_size=self.nucl_size_px, 
                          matrix_size=self.matrix_size
                         )
        
        bbox_tiler = tiler.mosaic_split()
        return bbox_tiler

    def preprocess_image(self, image):
        image = image[...,::-1]
        z = Normalize(max_pixel_value=255.)
        image = z(image=image)['image']
        return image

    def run(self, image, bboxes):
        self.image = np.copy(image)
        self.bboxes = bboxes
        self.inference_image = self.preprocess_image(self.image)
        self.image_tiler = self.tile_image_bbox(self.inference_image, self.bboxes)
        self.output_result = self.onnx_tile_inference(self.model, self.image_tiler)
        self.output_result = self.postprocess(self.output_result)
        return self.output_result

    @st.cache(allow_output_mutation=True)
    def load_onnx_model(self, model_path, cuda):
        if os.path.exists(model_path):
            providers = ['CUDAExecutionProvider'] if cuda else ['CPUExecutionProvider']
            session = ort.InferenceSession(model_path, providers=providers)  
        else:
            st.error("Model file does not exist")
        return session

    def onnx_tile_inference(self, session, tiler_loader):
        outname = [i.name for i in session.get_outputs()]
        inname = [i.name for i in session.get_inputs()]
        result = []
        # Run predictions for tiles and accumulate them
        for tiles_batch, source_bbox_batch, transformed_bbox_batch in stqdm(tiler_loader):
            tiles_batch = tiles_batch.transpose(2,0,1)
            tiles_batch = tiles_batch[np.newaxis,...]
            inp = {inname[0]:np.array(tiles_batch)}
            output = session.run(outname, inp)[0][0]
            output = np.argmax(output, 0)
            result.append([output, source_bbox_batch, transformed_bbox_batch])
        return result

    def postprocess(self, output):
        self.contours = []
        for batch in output:
            prediction, orig_bboxes, bboxes = batch
            for source, pred in zip(orig_bboxes, bboxes):
                source_bbx = [int(x) for x in source[1:5]]
                pred = [int(x) for x in pred]
                source = [int(x) for x in source]
                crop = prediction[pred[1]:pred[3],pred[0]:pred[2]].astype(np.uint8)
                if crop.shape[:2] == (0, 0):
                    continue
                crop = cv2.resize(crop, [int(source_bbx[2]-source_bbx[0]), 
                                         int(source_bbx[3]-source_bbx[1])], 
                                         cv2.INTER_NEAREST)
                
                cnt,_ = cv2.findContours(crop, cv2.RETR_LIST, 
                                         cv2.CHAIN_APPROX_SIMPLE, 
                                         offset=(source_bbx[0],source_bbx[1]))
                self.contours.append([cnt, source[-2]])
        return self.contours

    def plot_contours(self, cnts, image, plot_celltypes, fulfill):
        plot_dict = {c:plot_celltypes[c] for c in range(len(plot_celltypes))}
        edge = 3 if not fulfill else -1

        plt_img = image.copy()
        for [cnt, cls_id] in cnts:
            cls_id = int(cls_id)
            if not plot_dict[cls_id]:
                continue
            name = NAMES_DICT[cls_id]
            color = COLORS[name]
            cv2.drawContours(plt_img, cnt, -1, color, edge)
        return plt_img