import argparse
import cv2 
import streamlit as st
import plotly.express as px

from modules.gui import SidebarGUI
from modules.data import DataStore
from modules.detection_inference import ObjectDetection
from modules.segmentation_inference import ObjectSegmentation


def main(detection, segmentation):

    sidebar_gui = SidebarGUI()
    init_sidebar_params = sidebar_gui.get_gui_sidebar_params()
    data_store = DataStore(init_sidebar_params)
    st.session_state.image = data_store.load_image()

    if st.sidebar.button('Process image'):
        st.session_state.bboxes = detection.run(st.session_state.image)
        if init_sidebar_params['segmentation']:
            st.session_state.contours = segmentation.run(st.session_state.image, st.session_state.bboxes)

    if 'bboxes' not in st.session_state:
        return

    if init_sidebar_params['statistics']:
        counts_dict, props_dict = detection.calculate_subtypes(st.session_state.bboxes)
        st.write(f"Total count of cells: {len(st.session_state.bboxes)}")
        for name, val in counts_dict.items():
            st.text(f'{name}: {val} ({props_dict[name]}%)')

    plot_classes = sidebar_gui.postprocess_sidebar()

    detection_transparancy = st.slider("Transparancy (bbox)", min_value=0., max_value=1., value=0.75)
    fulfill_bbox = st.checkbox('Fulfill bbox', value=False)
    st.session_state.plt_bb_image = st.session_state.image.copy()
    st.session_state.plot_bb_img = detection.plot_outputs(st.session_state.bboxes, 
                                                       st.session_state.image, 
                                                       plot_classes, 
                                                       fulfill=fulfill_bbox)

    cv2.addWeighted(st.session_state.plot_bb_img, 
                    detection_transparancy, 
                    st.session_state.plt_bb_image, 
                    1-detection_transparancy,
                    0, 
                    st.session_state.plt_bb_image)

    fig = px.imshow(st.session_state.plt_bb_image)
    fig.update_layout(height=800)
    st.plotly_chart(fig, height=800)

    if 'contours' not in st.session_state:
        return
    segmentation_transparancy = st.slider("Transparancy (contour)", min_value=0., max_value=1., value=0.75)
    fulfill_cnts = st.checkbox('Fulfill contours', value=False)

    st.session_state.plt_cnt_image = st.session_state.image.copy()
    st.session_state.plot_cnt_img = segmentation.plot_contours(st.session_state.contours, 
                                                    st.session_state.image, 
                                                    plot_classes, 
                                                    fulfill=fulfill_cnts)
    cv2.addWeighted(st.session_state.plot_cnt_img, 
                    segmentation_transparancy, 
                    st.session_state.plt_cnt_image, 
                    1-segmentation_transparancy,
                    0, 
                    st.session_state.plt_cnt_image)

    fig = px.imshow(st.session_state.plt_cnt_image)
    fig.update_layout(height=800)
    st.plotly_chart(fig, height=800)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--detection_model', default='models/detection/best_reparametrize.onnx')
    parser.add_argument('-s', '--segmentation_model', default='models/segmentation/nuclei_segmenter_effnet_128.onnx')

    args = parser.parse_args()
    detection_path = args.detection_model
    segmentation_path = args.segmentation_model

    st.title("H&E nuclei multiclass workflow")
    st.subheader("Tool for nuclei detection and segmentation")
    st.write("Classes: tumor, immune cells, stromal cells, necrotic cells, non tumor cells")
    st.write("Default pixel size - 0.25 x 0.25 (um). If picture has lower resolution result may has lower accuracy")

    detection = ObjectDetection(model_path=detection_path)
    segmentation = ObjectSegmentation(model_path=segmentation_path)
    main(detection, segmentation)


