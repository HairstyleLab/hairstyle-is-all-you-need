import os
import sys
import cv2
import argparse
import numpy as np


def face_swap(src, dst):

    parser = argparse.ArgumentParser(description='FaceSwapApp')
    parser.add_argument('--warp_2d', default=False, action='store_true', help='2d or 3d warp')
    parser.add_argument('--correct_color', default=False, action='store_true', help='Correct color')
    args = parser.parse_args()

    src = os.path.abspath(src)
    dst = os.path.abspath(dst)

    original_cwd = os.getcwd()
    faceswap_dir = os.path.join(original_cwd, 'model/FaceSwap')

    try:
        os.chdir(faceswap_dir)
        sys.path.append(faceswap_dir)
        from model.FaceSwap.face_detection import select_face, select_all_faces
        from model.FaceSwap.face_swap import face_swap as swap_face
    finally:
        os.chdir(original_cwd)

    src_img = cv2.imread(src)
    dst_img = cv2.imread(dst)

    src_points, src_shape, src_face = select_face(src_img)

    dst_faceBoxes = select_all_faces(dst_img)

    if dst_faceBoxes is None:
        print('Detect 0 Face !!!')
        return dst_img

    output = dst_img

    for k, dst_face in dst_faceBoxes.items():
        output = swap_face(src_face, dst_face["face"], src_points,
                            dst_face["points"], dst_face["shape"],
                            output, args)

    _, img_encoded = cv2.imencode('.jpg', output)
    return img_encoded.tobytes()