import json
import base64
import random
from pathlib import Path
from openai import OpenAI

client = OpenAI()


def load_images_as_base64(image_folder: str) -> list:
    """이미지 폴더에서 모든 이미지를 base64로 로드"""
    
    image_folder = Path(image_folder)
    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    
    images = []
    for img_path in image_folder.iterdir():
        if img_path.suffix.lower() in image_extensions:
            with open(img_path, "rb") as f:
                img_base64 = base64.b64encode(f.read()).decode("utf-8")
            
            mime_type = "image/jpeg" if img_path.suffix.lower() in {".jpg", ".jpeg"} else f"image/{img_path.suffix[1:].lower()}"
            
            images.append({
                "filename": img_path.name,
                "base64": f"data:{mime_type};base64,{img_base64}"
            })
    
    print(f"로드된 이미지: {len(images)}개 ({image_folder})")
    return images


def generate_image_recommendation_exception_queries(num_samples: int = 50) -> list:
    """이미지 있는 추천 예외처리 질의 생성"""
    
    prompt = f"""
헤어스타일 추천 챗봇의 학습 데이터를 생성해주세요.

[시나리오]
사용자가 사진을 업로드하고 헤어스타일 추천을 요청하는 경우입니다.
하지만 이미지에 문제가 있어서 예외처리가 필요한 상황입니다.

[예외 케이스 2가지]

1. 얼굴 없는 이미지 (풍경, 음식, 동물, 물체 등)
   → 응답: "얼굴이 포함된 이미지를 첨부하셔야 헤어스타일 추천이 가능합니다🥲 확인 후 다른 사진을 업로드해주세요."

2. 얼굴 2명 이상 이미지 (단체사진, 커플사진 등)
   → 응답: "이 이미지에는 2명 이상의 얼굴이 포함되어 있습니다🥲 한 명만 나온 이미지를 업로드해주세요."

[생성 규칙]
각 케이스별로 다양한 질의를 생성해주세요:
- 케이스1 (얼굴 없음): {num_samples // 2}개
- 케이스2 (2명 이상): {num_samples // 2}개

다양한 표현 사용:
- "이 사진으로", "내 사진에", "이 얼굴에", "이 이미지로"
- "추천해줘", "알려줘", "뭐가 어울려?", "어떤 게 좋아?"
- 반말, 존댓말, 이모지 등

[출력 형식]
JSON 배열로 출력. 각 항목:
{{
  "type": "no_face" | "multi_face",
  "user": "사용자 질의"
}}

JSON 배열만 출력하고 다른 설명은 하지 마세요.
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,
    )
    
    # JSON 파싱
    content = response.choices[0].message.content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1]
    if content.endswith("```"):
        content = content.rsplit("```", 1)[0]
    
    samples = json.loads(content)
    return samples


# 고정 응답 매핑
RESPONSE_MAP = {
    "no_face": "얼굴이 포함된 이미지를 첨부하셔야 헤어스타일 추천이 가능합니다🥲 확인 후 다른 사진을 업로드해주세요.",
    "multi_face": "이 이미지에는 2명 이상의 얼굴이 포함되어 있습니다🥲 한 명만 나온 이미지를 업로드해주세요."
}


def convert_to_training_format(samples: list, no_face_images: list, multi_face_images: list) -> list:
    """생성된 샘플을 학습 데이터 형식으로 변환"""
    
    training_data = []
    
    for sample in samples:
        sample_type = sample["type"]
        
        # 타입에 따라 이미지 선택
        if sample_type == "no_face":
            selected_image = random.choice(no_face_images)
        else:  # multi_face
            selected_image = random.choice(multi_face_images)
        
        # 고정 응답 사용
        response = RESPONSE_MAP[sample_type]
        
        training_sample = {
            "messages": [
                {"role": "system", "content": ""},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": sample["user"]},
                        {"type": "image_url", "image_url": {"url": selected_image["base64"]}}
                    ]
                },
                {"role": "assistant", "content": response}
            ]
        }
        training_data.append(training_sample)
    
    return training_data


def save_to_jsonl(data: list, filename: str):
    """JSONL 파일로 저장"""
    with open(filename, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"저장 완료: {filename} ({len(data)}개 샘플)")


def get_data(
    no_face_folder: str = "images/no_face",
    multi_face_folder: str = "images/multi_face",
    num_samples: int = 50,
    output_file: str = "image_recommendation_exception.jsonl"
):
    """메인 함수: 이미지 로드 → 데이터 생성 → 변환 → 저장"""
    
    # 1. 이미지 로드
    print("이미지 로드 중...")
    no_face_images = load_images_as_base64(no_face_folder)
    multi_face_images = load_images_as_base64(multi_face_folder)
    
    if not no_face_images:
        raise ValueError(f"얼굴 없는 이미지가 없습니다: {no_face_folder}")
    if not multi_face_images:
        raise ValueError(f"다중 얼굴 이미지가 없습니다: {multi_face_folder}")
    
    # 2. GPT로 질의 생성
    print(f"이미지 추천 예외처리 샘플 {num_samples}개 생성 중...")
    raw_samples = generate_image_recommendation_exception_queries(num_samples)
    print(f"생성 완료: {len(raw_samples)}개")
    
    # 타입별 개수 확인
    type_counts = {}
    for sample in raw_samples:
        t = sample["type"]
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f"타입별 분포: {type_counts}")
    
    # 3. 학습 데이터 형식으로 변환
    training_data = convert_to_training_format(raw_samples, no_face_images, multi_face_images)
    
    # 4. JSONL 저장
    save_to_jsonl(training_data, output_file)
    
    return training_data


if __name__ == "__main__":
    data = get_data(
        no_face_folder="images/no_face",        # 풍경, 음식, 동물 등
        multi_face_folder="images/multi_face",  # 단체사진, 커플사진 등
        num_samples=50,
        output_file="samples/image_recommendation_exception.jsonl"
    )
    
    # 확인용 출력
    print("\n=== 샘플 미리보기 ===")
    for i, sample in enumerate(data[:4]):
        user_content = sample['messages'][1]['content']
        text = user_content[0]['text']
        img_preview = user_content[1]['image_url']['url'][:50] + "..."
        assistant_msg = sample['messages'][2]['content']
        
        print(f"\n[{i+1}] User: {text}")
        print(f"    Image: {img_preview}")
        print(f"    Assistant: {assistant_msg[:40]}...")

## 폴더 구조
"""
project/
├── images/
│   ├── normal_faces/    # 정상 얼굴 1명 (툴 호출용)
│   ├── no_face/         # 얼굴 없는 이미지 (풍경, 음식, 동물 등)
│   │   ├── landscape1.jpg
│   │   ├── food1.jpg
│   │   ├── animal1.jpg
│   │   └── ...
│   └── multi_face/      # 얼굴 2명 이상 (단체사진, 커플 등)
│       ├── group1.jpg
│       ├── couple1.jpg
│       └── ...
├── samples/
│   └── image_recommendation_exception.jsonl
└── generate_image_rec_exception.py


## 실행 결과 예시

이미지 로드 중...
로드된 이미지: 10개 (images/no_face)
로드된 이미지: 8개 (images/multi_face)
이미지 추천 예외처리 샘플 50개 생성 중...
생성 완료: 50개
타입별 분포: {'no_face': 25, 'multi_face': 25}
저장 완료: samples/image_recommendation_exception.jsonl (50개 샘플)

=== 샘플 미리보기 ===

[1] User: 이 사진으로 어울리는 헤어스타일 추천해줘
    Image: data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQAB...
    Assistant: 얼굴이 포함된 이미지를 첨부하셔야 헤어스타일 추천이...

[2] User: 내 사진에 맞는 머리 추천해줘~
    Image: data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQAB...
    Assistant: 얼굴이 포함된 이미지를 첨부하셔야 헤어스타일 추천이...

[3] User: 이 사진으로 헤어스타일 추천해주세요
    Image: data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQAB...
    Assistant: 이 이미지에는 2명 이상의 얼굴이 포함되어 있습니다🥲...

[4] User: 나한테 어울리는 머리 뭐야?
    Image: data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQAB...
    Assistant: 이 이미지에는 2명 이상의 얼굴이 포함되어 있습니다🥲...
"""