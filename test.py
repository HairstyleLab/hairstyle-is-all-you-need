import os
from model.utils import load_hairfastgan, generate_hairstyle, load_identiface, get_face_shape_and_gender

file_path = "images/face2.jpg"
face_img = "images/ky.jpg"
shape_img = "images/ew.jpg"
color_img = "images/gd.jpg"

model = load_identiface()
predicted_shape, shape_probs = get_face_shape_and_gender(model, file_path)

model = load_hairfastgan()
result = generate_hairstyle(model, face_img, shape_img, color_img)