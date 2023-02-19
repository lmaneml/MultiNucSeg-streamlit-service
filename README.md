MultiNucSeg is a worflow for multiclass nuclei detection and segmentation of H&E images. 
This repository includes streamlit web service. Inference is based on ONNX format.
For correct work it requires two models:
* Multiclass nuclei detection 
* Nuclei segmentation
Segmentation module works with mosaic bboxes, i.e. input image is mosaic nuclei crops from detection module.

Interface example:

<div align="center">
  <img src=" "/>
</div>
<br>

## How to run

With default models:
```
streamlit run main.py 
```
With custom models:
```
streamlit run main.py -- --detection_model 'model_path' --segmentation_model 'model_path'
```

# Models will be published later
