import json

import cv2
import numpy as np
import streamlit as st


def load_metadata(path, cell_info=False):
    with open(path, 'r') as fd:
        d = json.load(fd)
        if cell_info:
            d = {int(k):v for k, v in d.items()}
    return d

DEMO_IMAGES = load_metadata('./configs/demo_images.json')


class DataStore:

    def __init__(self, gui_sidebar_params):
        self.image_source = gui_sidebar_params['image_source']
        self.demo_images = DEMO_IMAGES
        if self.image_source == 'Demo':
            demo_type = gui_sidebar_params['demo_type']
            self.demo_image_path = self.demo_images[demo_type]

        self.image = None

    def load_image(self):   
        if self.image_source == 'Demo':
            @st.cache(allow_output_mutation=True)
            def load_demo_image(image_path):
                image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                return image
            self.image = load_demo_image(self.demo_image_path)

        elif self.image_source == 'Upload':

            @st.cache(allow_output_mutation=True)
            def load_external_image(image_file):
                image = image_file.getvalue()
                image = np.asarray(bytearray(image), dtype="uint8")
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                if image.shape[-1] > 3:
                    image = image[...,:3] 
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                return image

            image_file = st.sidebar.file_uploader(
                        'Upload H&E image, allowed formats: png, jp(e)g', 
                        type=['jpg', 'png', 'jpeg'],
                        accept_multiple_files=False, key=None, help='H&E image')
            if image_file:
                self.image = load_external_image(image_file)
        
        else:
            raise ValueError('Choose "Demo" or "Upload" image source')

        return self.image