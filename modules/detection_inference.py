import os
import math
import json

import cv2
import numpy as np
import streamlit as st
import onnxruntime as ort
from stqdm import stqdm

from .tiler import ImageSlicer
from .nms import non_max_suppression_fast


def load_metadata(path, cell_info=False):
    with open(path, 'r') as fd:
        d = json.load(fd)
        if cell_info:
            d = {int(k):v for k, v in d.items()}
    return d

CELLS_DICT = load_metadata('./configs/classes.json', cell_info=True)
NAMES_DICT = {x['idx']:x['name'] for x in CELLS_DICT.values()}
COLORS = {x['name']:x['color'] for x in CELLS_DICT.values()}

class ObjectDetection:

    def __init__(self, model_path):
        self.tile_size = [480, 480]
        self.tile_step = [360, 360]
        self.nms_threshold = 0.15
        self.model_path = model_path
        self.model = self.load_onnx_model(self.model_path, cuda=False)

        self.output_result = []

    def tile_image(self, image):
        slicer = ImageSlicer(image_shape=image.shape, 
                                  tile_size=self.tile_size, 
                                  tile_step=self.tile_step)
        image_tiler = slicer.iter_split(image)
        return image_tiler
        
    def preprocess_image(self, image):
        image = image.astype(np.float32)
        image /= 255
        return image

    def run(self, image):
        self.inference_image = np.copy(image)
        image_shape = image.shape
        dx, dy = self.calc_dx_dy(image_shape, self.tile_size, self.tile_step)
        self.inference_image = self.preprocess_image(self.inference_image)
        self.image_tiler = self.tile_image(self.inference_image)
        self.output_result = self.onnx_tile_inference(self.model, self.image_tiler)
        self.output_result = self.rescale_coords(self.output_result, dx, dy, image_shape)
        self.output_result = non_max_suppression_fast(self.output_result, self.nms_threshold)
        return self.output_result

    def calc_dx_dy(self, img_size, tile_size, tile_step):
        ''' Function to fix pytorch-toolbelt tiler shift'''
        image_height, image_width = img_size[0], img_size[1]
        overlap = (tile_size[0] - tile_step[0], tile_size[1] - tile_step[1])

        margin_left = 0
        margin_right = 0
        margin_top = 0
        margin_bottom = 0

        nw = max(1, math.ceil((image_width - overlap[1]) / tile_step[1]))
        nh = max(1, math.ceil((image_height - overlap[0]) / tile_step[0]))

        extra_w = tile_step[1] * nw - (image_width - overlap[1])
        extra_h = tile_step[0] * nh - (image_height - overlap[0])

        margin_left = extra_w // 2
        margin_right = extra_w - margin_left
        margin_top = extra_h // 2
        margin_bottom = extra_h - margin_top
        dx, dy = margin_right, margin_bottom
        
        return dx, dy

    def rescale_coords(self, coords, dx, dy, img_shape):
        ymax, xmax = img_shape[0], img_shape[1]
        drop_list = []
        counter = 0
        for batch in coords:
            for coord in batch[0]:
                x0, y0 = batch[1][0], batch[1][1]
                coord[2], coord[4] = coord[2]+y0-dy, coord[4]+y0-dy
                coord[1], coord[3] = coord[1]+x0-dx, coord[3]+x0-dx
                coord[1:5] = [max(g,0) for g in coord[1:5]]
                coord[1] = min(coord[1], xmax)
                coord[3] = min(coord[3], xmax)
                coord[2] = min(coord[2], ymax)
                coord[4] = min(coord[4], ymax)
                if int(coord[1])==int(coord[3]) or int(coord[2])==int(coord[4]):
                    drop_list.append(counter)
                counter += 1
        coords = np.vstack([x[0] for x in coords])
        coords = np.delete(coords, drop_list, 0)
        return coords

    @st.cache(allow_output_mutation=True)
    def load_onnx_model(self, model_path, cuda):
        if os.path.exists(model_path):
            providers = ['CUDAExecutionProvider'] if cuda else ['CPUExecutionProvider']
            session = ort.InferenceSession(model_path, providers=providers)  
        else:
            st.error("Model file does not exist")
        return session

    def onnx_inference(self, session, image):
        outname = [i.name for i in session.get_outputs()]
        inname = [i.name for i in session.get_inputs()]
        outputs = session.run(outname, {inname[0]:image})[0]
        return outputs

    def onnx_tile_inference(self, session, tiler_loader):
        outname = [i.name for i in session.get_outputs()]
        inname = [i.name for i in session.get_inputs()]
        result = []
        for tiles_batch, coords_batch in stqdm(tiler_loader):
            tiles_batch = tiles_batch.transpose(2,0,1)
            tiles_batch = tiles_batch[np.newaxis,...]
            inp = {inname[0]:np.array(tiles_batch)}
            outputs = session.run(outname, inp)[0]
            result.append([outputs, coords_batch])
        return result

    def plot_bbox(self, bbox_row, image, fulfill):
        edge = 3 if not fulfill else -1
        batch_id,x0,y0,x1,y1,cls_id,score = bbox_row
        cls_id = int(cls_id)
        box = [int(round(z)) for z in [x0,y0,x1,y1]]
        color = self.COLORS[cls_id]
        cv2.rectangle(image,box[:2],box[2:],color,edge)


    def plot_outputs(self, bboxes, image, plot_celltypes, fulfill=1):
        plot_dict = {c:plot_celltypes[c] for c in range(len(plot_celltypes))}
        edge = 3 if not fulfill else -1
        plt_img = image.copy()
        for batch_id,x0,y0,x1,y1,cls_id,score in bboxes:
            cls_id = int(cls_id)
            if not plot_dict[cls_id]:
                continue
            box = np.array([x0,y0,x1,y1])
            box = box.round().astype(np.int32).tolist()
            # score = round(float(score),3)
            name = NAMES_DICT[cls_id]
            color = COLORS[name]
            cv2.rectangle(plt_img,box[:2],box[2:],color,edge)
        return plt_img

    def calculate_subtypes(self, bboxes):
        counts = bboxes[:,5]
        counts = np.bincount(counts, minlength=len(NAMES_DICT.keys()))
        total_number = len(bboxes)
        props = [x/total_number for x in counts]
        counts_dict = {}
        props_dict = {}
        for i in range(len(NAMES_DICT.keys())):
            name = NAMES_DICT[i]
            counts_dict[name] = counts[i]
            props_dict[name] = round(props[i]*100, 2)
        return counts_dict, props_dict