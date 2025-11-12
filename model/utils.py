import os
import tensorflow as tf
from model.HairFastGAN.hair_swap import HairFast, get_parser
from model.IdentiFace.Backend.model_manager import model_manager
from model.IdentiFace.Backend.functions import Functions

def load_hairfastgan():
    model = HairFast(get_parser().parse_args([]))

    return model

def generate_hairstyle(model, face_img, shape_img, color_img):
    result = model(face_img, shape_img, color_img)

    return result

def load_identiface():
    print("Loading models...")
    with tf.device('/CPU:0'):
        model_manager.load_models()
    print("Models loaded.")

    return model_manager

def get_face_shape_and_gender(model, file_path):
    if not os.path.isfile(file_path):
        print("File does not exist!")
        return

    result = Functions.preprocess("offline", file_path)
    if result is None:
        print("Error preprocessing image.")
        return
    path, normalized_face = result

    if model.shape_model is not None and model.gender_model is not None:
        with tf.device('/CPU:0'):
            predicted_shape, shape_probs = Functions.predict_shape("offline", file_path, model.shape_model)
            predicted_gender, gender_probs = Functions.predict_gender("offline", file_path, model.gender_model)
        print(f"Predicted Shape: {predicted_shape}")
        print(f"Predicted gneder: {predicted_gender}")
    else:
        print("Shape model or Gender model not loaded.")

    return predicted_shape, predicted_gender