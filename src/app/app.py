import streamlit as st
import base64
import torch
from torch import nn
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import io
import os

from ai_hack.models import Embedder
from transformers import ViTImageProcessor, ViTModel

import faiss
import pickle


def convert_png_to_jpg(png_file):
    img = Image.open(png_file)
    img = img.convert('RGB')
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    return buffer


def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = """
    <style>
        #root > div:nth-child(1) > div > div > div > div > section > div {padding-top: 0rem;}
        [data-testid="stAppViewContainer"] {
            background-image: url("data:image/png;base64,%s");
            background-attachment: fixed;
            background-repeat: no-repeat;
            background-position: left 100px;
            color: #000000;
        }
    </style>
    """ % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)


set_png_as_page_bg('eagle_blur.png')


st.title("AI Integration")
on = st.toggle("Скриншот")

from config import *
dimension = 64

device = "cuda" if torch.cuda.is_available() else "cpu"

image_processor = ViTImageProcessor.from_pretrained('facebook/dino-vits16')
trunk = ViTModel.from_pretrained('facebook/dino-vits16')
trunk_output_size = trunk.config.hidden_size
trunk = nn.DataParallel(trunk).to(device)
embedder = nn.DataParallel(Embedder(input_dim=trunk_output_size,
                                        embedding_dim=dimension)).to(device)
models = {'trunk': trunk, 'embedder': embedder}
checkpoint = torch.load(MODEL_CHECKPOINT_PATH, map_location=torch.device(device))
models['trunk'].load_state_dict(checkpoint['trunk_state_dict'])
models['embedder'].load_state_dict(checkpoint['embedder_state_dict'])


index = faiss.read_index(FAISS_INDEX_PATH)

# Загрузка словаря путей изображений
with open(IMAGE_IDX_MAP_PATH, 'rb') as f:
    image_paths = pickle.load(f)

def get_embedding(image):
    with torch.no_grad():
        image = image_processor(images=image, return_tensors='pt')['pixel_values'].squeeze(0)
        image_vit = trunk(image.unsqueeze(0))
        embedding = embedder(image_vit).cpu().numpy()
    return embedding



def get_copies(image, image_paths):
    k = 10  # количество ближайших соседей
    distances, indices = index.search(get_embedding(image), k)
    indices = indices.flatten()
    im_p = [image_paths[idx] for idx in indices]
    return im_p


if on:
    st.header("Загрузите скриншот:")
    screen = st.file_uploader(label=" ")
    if screen:
        screen = convert_png_to_jpg(screen)
        screen.seek(0)
        st.image(screen)

        model_id = "IDEA-Research/grounding-dino-tiny"

        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
        image = Image.open(screen)

        text = "find colored images on the screen"

        inputs = processor(images=image, text=text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.06,
            text_threshold=0.16,
            target_sizes=[image.size[::-1]]
        )

        original_area = image.size[0] * image.size[1]
        selected_box = None
        max_score = -1

        for box, label, score in zip(results[0]['boxes'], results[0]['labels'], results[0]['scores']):
            x_min, y_min, x_max, y_max = [float(coord) for coord in box]
            box_area = (x_max - x_min) * (y_max - y_min)
    
            if box_area != original_area and score > max_score:
                selected_box = (x_min, y_min, x_max, y_max)
                max_score = score

        if selected_box:
            cropped_image = image.crop(selected_box)
            st.header("\nНайденное изображение:")
            left_co, cent_co,last_co = st.columns(3)
            with cent_co:
                st.image(cropped_image)


            st.header("\nРанжированные смысловые копии:")
            cols = st.columns(2)

            copies = get_copies(cropped_image, image_paths)

            i = 0
            for copy in copies:
                cols[i%2].image(copy)
                i += 1
        else:
            st.hear("\nИзображения не найдены.")

else:
    st.header("Загрузите образец:")
    pic = st.file_uploader(label="")
    if pic:
        st.image(pic)
        pic = convert_png_to_jpg(pic)
        image = Image.open(pic)
        st.header("\nРанжированные смысловые копии:")
        cols = st.columns(2)

        copies = get_copies(image, image_paths)

        i = 0
        for copy in copies:
            cols[i%2].image(copy)
            i += 1

