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

# 퍼스널 컬러 스킨 톤-2017년 논문 기반 분류 알고리즘
def srgb_to_linear(c):
    c = c / 255.0
    if c <= 0.04045:
        return c / 12.92
    else:
        return ((c + 0.055) / 1.055) ** 2.4

def f_lab(t):
    delta = 6/29
    if t > delta**3:
        return t ** (1/3)
    else:
        return t / (3 * delta**2) + 4/29

def rgb_tuple_to_lab(rgb_tuple):
    r, g, b = rgb_tuple

    r_lin = srgb_to_linear(r)
    g_lin = srgb_to_linear(g)
    b_lin = srgb_to_linear(b)

    y = r_lin * 0.2126 + g_lin * 0.7152 + b_lin * 0.0722
    z = r_lin * 0.0193 + g_lin * 0.1192 + b_lin * 0.9505
    
    fy = f_lab(y / 1.00000)
    fz = f_lab(z / 1.08883)

    L = 116 * fy - 16
    b = 200 * (fy - fz)

    return L,b

def classify_personal_color(rgb_tuple):
    # 논문에서 제시한 기준값을 실험적으로 조절
    V0 = 72.5     
    B0 = 15.5     
    V, bstar = rgb_tuple_to_lab(rgb_tuple)

    if bstar >= B0:
        if V >= V0:
            return "봄웜"
        else:
            return "가을웜"
    else:
        if V >= V0:
            return "여름쿨"
        else:
            return "겨울쿨"