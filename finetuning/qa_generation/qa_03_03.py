import json
import random
import base64
from pathlib import Path
from openai import OpenAI

client = OpenAI()


def load_images_as_base64(folder: str):
    folder = Path(folder)
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    images = []

    for p in folder.iterdir():
        if p.suffix.lower() in exts:
            with open(p, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            mime = "image/jpeg" if p.suffix.lower() in [".jpg", ".jpeg"] else f"image/{p.suffix[1:].lower()}"
            images.append({
                "filename": p.name,
                "base64": f"data:{mime};base64,{b64}"
            })

    print(f"[로드] {folder} → {len(images)}개")
    return images


def generate_exception_queries(num_samples=60):
    """
    4가지 케이스 균등 생성:
      - no_face
      - multi_face
      - unsupported_style
      - missing_style
    """

    prompt = f"""
다음 4가지 케이스에 해당하는 이미지 생성 예외처리용 사용자 질의를 생성하세요.

[케이스 종류]
1) no_face (얼굴 없음)
2) multi_face (2명 이상)
3) unsupported_style (존재하지 않는 스타일 요청)
4) missing_style (스타일/컬러 미지정)

[생성 규칙]
- 총 {num_samples}개 생성
- 각 타입별 동일 개수 생성
- 반말/존댓말/이모지 섞기
- 표현 다양하게: "이 사진에", "내 이미지로", "이 얼굴로", "머리 바꿔줘", "스타일 적용해줘" 등

[출력 형식: JSON 배열만]
각 항목:
{{
  "type": "no_face" | "multi_face" | "unsupported_style" | "missing_style",
  "user": "사용자 질의"
}}
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,
    )

    content = response.choices[0].message.content.strip()

    if content.startswith("```"):
        content = content.split("\n", 1)[1]
    if content.endswith("```"):
        content = content.rsplit("```", 1)[0]

    return json.loads(content)


RESPONSE_MAP = {
    "no_face":
        "얼굴이 포함된 이미지를 첨부하셔야 이미지를 만들 수 있습니다🥲 확인 후 다른 사진을 업로드해주세요.",

    "multi_face":
        "이 이미지에는 2명 이상의 얼굴이 포함되어 있습니다🥲 한 명만 나온 이미지를 업로드해주세요.",

    "unsupported_style":
        "죄송합니다. 요청하신 스타일은 지원되지 않는 스타일입니다.\n\n"
        "지원되는 여자 펌: C컬펌, S컬펌, 글램펌, 내츄럴펌, 러블리펌, 루즈펌, 리프펌, 물결펌, 바디펌, 발롱펌, "
        "볼드펌, 볼륨매직, 볼륨펌, 빌드펌, 에어펌, 젤리펌, 지젤펌, 쿠션펌, 텍스처펌, 퍼피베이비펌, 허쉬펌_롱\n\n"
        "위 목록에서 원하시는 스타일을 선택해주세요😊",

    "missing_style":
        "어떤 헤어스타일이나 헤어컬러로 변경하고 싶으신가요? 원하시는 스타일이나 컬러를 말씀해주세요😊"
}


def convert_to_training_format(samples, no_face_images, multi_face_images, normal_images):

    training_data = []

    for s in samples:
        stype = s["type"]

        # 이미지 선택
        if stype == "no_face":
            img = random.choice(no_face_images)
        elif stype == "multi_face":
            img = random.choice(multi_face_images)
        else:
            img = random.choice(normal_images)  # unsupported, missing → 정상 얼굴

        assistant_reply = RESPONSE_MAP[stype]

        training_data.append({
            "messages": [
                {"role": "system", "content": ""},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": s["user"]},
                        {"type": "image_url", "image_url": {"url": img["base64"]}}
                    ]
                },
                {"role": "assistant", "content": assistant_reply}
            ]
        })

    return training_data


def save_jsonl(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[저장 완료] {filename} ({len(data)}개)")


def build_img2img_exception_dataset(
    no_face_folder="images/no_face",
    multi_face_folder="images/multi_face",
    normal_face_folder="images/normal_faces",
    num_samples=60,
    output="samples/image_gen_exception.jsonl",
):

    print("\n### 이미지 로딩")
    no_face_imgs = load_images_as_base64(no_face_folder)
    multi_face_imgs = load_images_as_base64(multi_face_folder)
    normal_imgs = load_images_as_base64(normal_face_folder)

    if not no_face_imgs or not multi_face_imgs or not normal_imgs:
        raise ValueError("이미지가 충분하지 않습니다!")

    print("\n### GPT 사용자 질의 생성")
    raw_samples = generate_exception_queries(num_samples)
    print(f"[생성 완료] {len(raw_samples)}개")

    print("\n### 학습 포맷 변환")
    training_data = convert_to_training_format(
        raw_samples,
        no_face_imgs,
        multi_face_imgs,
        normal_imgs
    )

    print("\n### JSONL 저장")
    save_jsonl(training_data, output)

    return training_data


if __name__ == "__main__":
    data = build_img2img_exception_dataset(
        no_face_folder="images/no_face",
        multi_face_folder="images/multi_face",
        normal_face_folder="images/normal_faces",
        num_samples=80,
        output="samples/image_gen_exception.jsonl"
    )
