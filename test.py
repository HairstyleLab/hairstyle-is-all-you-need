import os
from model.utils import load_identiface, get_face_shape_and_gender

file_path = "images/face2.jpg"

model = load_identiface()
predicted_shape, shape_probs = get_face_shape_and_gender(model, file_path)