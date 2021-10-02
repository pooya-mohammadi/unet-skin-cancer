import os
import sys
import glob
import numpy as np
import streamlit as st
from PIL import Image
import base64
from io import BytesIO
from config import MODELS_TO_ADDR, MODELS_TO_ARGS

sys.path.append(os.path.split(os.path.split(__file__)[0])[0])
from detect import Detect


def get_image_download_link(input_img, filename="result.png", text="Download result"):
    img = Image.fromarray(input_img)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'
    return href


# page config
st.set_page_config(
    page_title="AIMedic",
    page_icon='files/aimedic.png')

# sidebar header
header_col1, header_col2 = st.sidebar.columns((2, 7))
header_col1.image(Image.open('streamlit/files/aimedic.png'), use_column_width=True)
header_col2.title("AIMedic")
st.sidebar.title("Skin-Cancer Segmentation")

# select model
models_option = st.sidebar.selectbox(
    'Models:',
    MODELS_TO_ADDR.keys()
)

# select image
st.sidebar.header("Input Image")
uploaded_file = st.sidebar.file_uploader(
    'upload cell image',
    type=['png', 'jpg', 'jpeg'],
    accept_multiple_files=False)

select_image_col1, select_image_col2 = st.sidebar.columns(2)
use_random = select_image_col1.button("Random Image")

if (use_random):
    random_images = glob.glob("streamlit/files/random-images/*")
    img = np.random.choice(random_images)
    img = np.array(Image.open(img))
    st.session_state['image'] = img
elif uploaded_file:
    img = np.array(Image.open(uploaded_file))
    st.session_state['image'] = img

# process image
process_btn = None
if 'image' in st.session_state:
    process_btn = select_image_col2.button("Process Image")

# page body
st.markdown('**Skin Cancer Segmentation** with **``%s``**' % models_option)
body_col1, body_col2 = st.columns(2)

if 'image' in st.session_state:
    body_col1.write("Input Image:")
    body_col1.image(st.session_state['image'], use_column_width=True)


@st.cache
def get_detector(model_name, weight_path, **kwargs):
    return Detect(model_name=model_name, weight_path=weight_path, **kwargs)


if process_btn:
    body_col2.write("Result Image:")
    detector = get_detector(
        models_option,
        MODELS_TO_ADDR[models_option],
        **MODELS_TO_ARGS[models_option]
    )
    result_image = detector.detect(st.session_state['image'])
    result_image = np.array(result_image)
    body_col2.image(result_image, use_column_width=True)
    st.markdown(get_image_download_link(result_image), unsafe_allow_html=True)
