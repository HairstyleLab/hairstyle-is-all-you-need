import os
import sys
import torch
import base64
import statistics
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
# from model.HairFastGAN.hair_swap import HairFast, get_parser
from model.IdentiFace.Backend.model_manager import model_manager
from model.IdentiFace.Backend.functions import Functions
sys.path.append('model/FaceLift')
sys.path.append('model/face_parsing')
sys.path.append('model/SAFMN')
from model.FaceLift.inference import get_model_paths, initialize_face_detector, initialize_mvdiffusion_pipeline, initialize_gslrm_model, setup_camera_parameters, process_single_image
from model.utility.face_cropper import FaceCropper

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

def display_image(image_bytes):
    img = Image.open(BytesIO(image_bytes))

    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis("off")
    plt.show()

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
            return "봄 웜톤"
        else:
            return "가을 웜톤"
    else:
        if V >= V0:
            return "여름 쿨톤"
        else:
            return "겨울 쿨톤"
        
def get_faceshape(face_shape:str)->str:
    match face_shape:
        case "Round":
            return "둥근형"
        case "Oval":
            return "계란형"
        case "Heart":
            return "하트형"
        case "Oblong":
            return "긴형"
        case "Square":
            return "사각형"
        
def get_weight(max_value, min_value):
    if max_value >0 and min_value >0:
        weight = statistics.mean([max_value,min_value])
    elif max_value>0 and min_value<0:
        weight = max_value / 2
    else:
        weight = abs(max_value)
    return weight

def get_3d(
    image_file,
    input_dir: str = './generated_images/',
    output_dir: str = './3d_results/',
    auto_crop: bool = True,
    seed: int = 4,
    guidance_scale_2D: float = 3.0,
    step_2D: int = 50,
    models_3d: dict = None
) -> str:
    """
    Generate 3D reconstruction from image

    Args:
        image_file: Path to image file
        input_dir: Input directory
        output_dir: Output directory
        auto_crop: Whether to auto crop face
        seed: Random seed
        guidance_scale_2D: Guidance scale for 2D generation
        step_2D: Steps for 2D generation
        models_3d: Pre-loaded 3D models dict (if None, will load on-the-fly)

    Returns:
        Path to the generated .ply file
    """
    # If models are not provided, load them (backward compatibility)
    if models_3d is None:
        computation_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        mvdiffusion_checkpoint_path, gslrm_checkpoint_path, gslrm_config_path = get_model_paths()

        os.makedirs(output_dir, exist_ok=True)

        face_detector = None
        if auto_crop:
            face_detector = initialize_face_detector(computation_device)

        diffusion_pipeline, random_generator, color_prompt_embeddings = initialize_mvdiffusion_pipeline(
            mvdiffusion_checkpoint_path, computation_device
        )
        gslrm_model = initialize_gslrm_model(gslrm_checkpoint_path, gslrm_config_path, computation_device)
        camera_intrinsics_tensor, camera_extrinsics_tensor = setup_camera_parameters(computation_device)
    else:
        # Use pre-loaded models
        face_detector = models_3d['face_detector']
        diffusion_pipeline = models_3d['diffusion_pipeline']
        random_generator = models_3d['random_generator']
        color_prompt_embeddings = models_3d['color_prompt_embeddings']
        gslrm_model = models_3d['gslrm_model']
        camera_intrinsics_tensor = models_3d['camera_intrinsics_tensor']
        camera_extrinsics_tensor = models_3d['camera_extrinsics_tensor']
        os.makedirs(output_dir, exist_ok=True)

    random_generator.manual_seed(seed)

    process_single_image(
        image_file,
        input_dir,
        output_dir,
        auto_crop,
        diffusion_pipeline,
        random_generator,
        color_prompt_embeddings,
        gslrm_model,
        camera_intrinsics_tensor,
        camera_extrinsics_tensor,
        guidance_scale_2D,
        step_2D,
        face_detector
    )

    # Generate the expected .ply file path
    image_name = image_file.split(".")[0]
    ply_path = os.path.join(output_dir, image_name, "gaussians.ply")

    return ply_path

def face_crop(image_file, crop_size=256, face_cropper=None):
    """
    Crop and process face from image

    Args:
        image_file: Path to image file
        crop_size: Size for face cropping
        face_cropper: Pre-loaded FaceCropper instance (if None, will load on-the-fly)

    Returns:
        Face tensor or None if multiple/no faces detected
    """
    # If face_cropper is not provided, load it (backward compatibility)
    if face_cropper is None:
        faceCropper = FaceCropper(crop_size)
    else:
        faceCropper = face_cropper

    image = Image.open(image_file).convert("RGB")
    image = smart_resize(image)

    detected_bboxs = faceCropper.faceDetector(image, True)
    numberDetectedFaces = len(detected_bboxs)

    print(f"[DEBUG] face_crop: 감지된 얼굴 수 = {numberDetectedFaces}")
    print(f"[DEBUG] detected_bboxs = {detected_bboxs}")

    if numberDetectedFaces == 1:
        face_crop_course, face_bbox_course, lmk68 = faceCropper.detect_face_simple(image)
        face_crop_refine, face_bbox_refine = faceCropper.refineCrop(face_crop_course, face_bbox_course)

        # Extend crop area vertically for longer hairstyles
        # face_bbox_refine: [start_y, end_y, start_x, end_x]
        height = face_bbox_refine[1] - face_bbox_refine[0]
        width = face_bbox_refine[3] - face_bbox_refine[2]

        # Add extra margin to top and bottom
        extra_top = int(height * 0.3)
        extra_bottom = int(height * 0.2)

        new_start_y = max(0, face_bbox_refine[0] - extra_top)
        new_end_y = min(np.array(image).shape[0], face_bbox_refine[1] + extra_bottom)

        # Re-crop from original image with extended vertical bounds
        face_crop_refine = np.array(image)[new_start_y:new_end_y, face_bbox_refine[2]:face_bbox_refine[3]]

    else:
        return None

    face_crop_refine = pad_to_square(face_crop_refine)
    face_tensor = np.array(face_crop_refine).astype(np.float32) / 255.0
    face_tensor = torch.from_numpy(np.transpose(face_tensor, (2, 0, 1))).unsqueeze(0)

    face_tensor = torch.nn.functional.interpolate(
        face_tensor,
        size=(512, 512),
        mode="bilinear",
        align_corners=False
    )

    return face_tensor

def pad_to_square(img):
    h, w, _ = img.shape
    if h == w:
        return img

    size = max(h, w)
    padded = np.zeros((size, size, 3), dtype=img.dtype)

    y_offset = (size - h) // 2
    x_offset = (size - w) // 2

    padded[y_offset:y_offset+h, x_offset:x_offset+w] = img
    return padded

def smart_resize(image):

  image = np.array(image)

  min_indx = np.argmin((image.shape[0], image.shape[1]))
  if image.shape[min_indx] > 512:

    dim1 = min(512, image.shape[min_indx])
    wpercent = image.shape[int(not min_indx)] / image.shape[int(min_indx)]
    dim2 = int(512 * wpercent)

    if min_indx == 0:
      image = np.array(Image.fromarray(image).resize((dim2, dim1), Image.LANCZOS, ))
    else:
      image = np.array(Image.fromarray(image).resize((dim1, dim2), Image.LANCZOS, ))

  return image
def encode_image_from_file(file_path):
    with open(file_path, "rb") as image_file:
        image_content = image_file.read()
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext in [".jpg", ".jpeg"]:
            mime_type = "image/jpeg"
        elif file_ext == ".png":
            mime_type = "image/png"
        else:
            mime_type = "image/unknown"
        return f"data:{mime_type};base64,{base64.b64encode(image_content).decode('utf-8')}"
