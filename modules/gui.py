import cv2
import streamlit as st


palette_path = './images/palette.png'

@st.cache(allow_output_mutation=True)
def load_palette(palette_path):
    palette = cv2.imread(palette_path, cv2.IMREAD_COLOR)
    palette = cv2.cvtColor(palette, cv2.COLOR_BGR2RGB)
    return palette

class SidebarGUI:
    
    def __init__(self):
        self.gui_sidebar_params = {}

    def get_gui_sidebar_params(self):
        return self.sidebar_init_config()

    def sidebar_init_config(self):
        st.sidebar.title("Prediction settings")
        st.sidebar.markdown("## Image source")

        self.image_source = st.sidebar.radio(
            "Select the source of image",
            ["Demo", "Upload"]
            )
        if self.image_source == 'Demo':
            self.demo_type = st.sidebar.radio('Select demo sample',
                ['Tumor (small FOV)', 'Tumor (big FOV)', 'Immune infiltrate', 'Stroma'])
        else:
            self.demo_type = None

        self.segmentation = st.sidebar.checkbox('Nuclei segmentation', value=False, help='Perform nuclei segmentation')
        self.statistics = st.sidebar.checkbox('Show statistics', value=False, help='Show counts of cell subtypes')

        self.gui_sidebar_params.update({
                                'image_source': self.image_source,
                                'demo_type': self.demo_type,
                                'segmentation': self.segmentation,
                                'statistics': self.statistics
                                })
        return self.gui_sidebar_params

    def postprocess_sidebar(self):
        st.sidebar.write("Visualization params")
        st.sidebar.write('Select classes:')
        tumor = st.sidebar.checkbox('Neoplastic cells', value=True)
        immune = st.sidebar.checkbox('Immune cells', value=True)
        stromal = st.sidebar.checkbox('Stromal cells', value=True)
        necro = st.sidebar.checkbox('Necrotic cells', value=True)
        non_tumor = st.sidebar.checkbox('Non-neoplastic epithelial cells', value=True)
        plot_classes =  [tumor, immune, stromal, necro, non_tumor]
        palette = load_palette(palette_path)
        st.sidebar.image(palette)
        return plot_classes