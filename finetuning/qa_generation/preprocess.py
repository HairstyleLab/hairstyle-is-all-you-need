# preprocess_images.py

import base64
import json
from pathlib import Path


def preprocess_images_to_base64(image_folder: str, output_json: str):
    image_folder = Path(image_folder)
    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}

    output = []

    for img_path in image_folder.iterdir():
        if img_path.suffix.lower() in image_extensions:
            with open(img_path, "rb") as f:
                img_base64 = base64.b64encode(f.read()).decode("utf-8")

            mime_type = (
                "image/jpeg"
                if img_path.suffix.lower() in {".jpg", ".jpeg"}
                else f"image/{img_path.suffix[1:].lower()}"
            )

            output.append({
                "filename": img_path.name,
                "base64": f"data:{mime_type};base64,{img_base64}"
            })

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"{len(output)}개 이미지 인코딩 완료 → {output_json}")


if __name__ == "__main__":
    preprocess_images_to_base64(
        "finetuning/images/normal_face",
        "finetuning/images/normal_face.json"
    )

    preprocess_images_to_base64(
        "finetuning/images/no_face",
        "finetuning/images/no_face.json"
    )

    preprocess_images_to_base64(
        "finetuning/images/multi_face",
        "finetuning/images/multi_face.json"
    )