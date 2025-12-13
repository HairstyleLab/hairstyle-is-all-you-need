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
                
            # 확장자에 따른 MIME 타입
            mime_type = "image/jpeg" if img_path.suffix.lower() in {".jpg", ".jpeg"} else f"image/{img_path.suffix[1:].lower()}"
            
            images.append({
                "filename": img_path.name,
                "base64": f"data:{mime_type};base64,{img_base64}"
            })
    
    print(f"로드된 이미지: {len(images)}개")
    return images


def generate_image_recommendation_samples(num_samples: int = 50) -> list:
    """이미지 있는 헤어스타일 추천 샘플 생성"""
    
    prompt = f"""
헤어스타일 추천 챗봇의 학습 데이터를 생성해주세요.

[시나리오]
사용자가 자신의 사진을 업로드하고 헤어스타일 추천을 요청하는 경우입니다.
이때 hairstyle_recommendation_tool을 호출해야 합니다.

[파라미터 추출 규칙]
1. keywords: 사용자가 원하는 스타일/컬러 관련 키워드 (시원한, 가벼운, 볼륨있는, 청순한, 밝은, 어두운 등)
   - 키워드가 없으면 빈 문자열 ""
2. season: 봄, 여름, 가을, 겨울 (언급된 경우만)

[생성할 질문 유형]
1. 단순 추천: "이 사진으로 어울리는 머리 추천해줘"
2. 키워드 포함: "이 얼굴에 어울리는 시원하고 가벼운 머리 추천해줘"
3. 계절 포함: "여름인데 어울리는 헤어스타일 추천해줘"
4. 키워드 + 계절: "겨울이라 따뜻해 보이는 볼륨있는 머리 추천해줘"
5. 염색 포함: "밝은 색으로 염색하고 싶은데 추천해줘"
6. 복합: "여름이라 시원하고 가벼운 느낌에 톤다운된 색으로 추천해줘"

다양한 말투로 {num_samples}개 생성해주세요.
- "이 사진", "내 사진", "이 얼굴", "내 얼굴" 등 다양한 표현 사용
- 반말, 존댓말, 이모지 등 다양하게

[출력 형식]
JSON 배열로 출력. 각 항목:
{{
  "user": "사용자 질의 (이미지 언급 포함)",
  "keywords": "키워드1, 키워드2" (없으면 ""),
  "season": "여름" (없으면 null)
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


def convert_to_training_format(samples: list, images: list) -> list:
    """생성된 샘플을 학습 데이터 형식으로 변환 (이미지 랜덤 선택)"""
    
    training_data = []
    
    for i, sample in enumerate(samples):
        # 랜덤 이미지 선택
        selected_image = random.choice(images)
        
        # arguments 구성
        arguments = {}
        if sample.get("keywords"):
            arguments["keywords"] = sample["keywords"]
        else:
            arguments["keywords"] = ""
        
        if sample.get("season"):
            arguments["season"] = sample["season"]
        
        arguments_str = json.dumps(arguments, ensure_ascii=False)
        
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
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call_{i+1:03d}",
                            "type": "function",
                            "function": {
                                "name": "hairstyle_recommendation_tool",
                                "arguments": arguments_str
                            }
                        }
                    ]
                }
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
    image_folder: str = "images/normal_faces",
    num_samples: int = 50,
    output_file: str = "image_recommendation.jsonl"
):
    """메인 함수: 이미지 로드 → 데이터 생성 → 변환 → 저장"""
    
    # 1. 이미지 로드
    print(f"이미지 폴더 로드 중: {image_folder}")
    images = load_images_as_base64(image_folder)
    
    if not images:
        raise ValueError(f"이미지가 없습니다: {image_folder}")
    
    # 2. GPT로 샘플 생성
    print(f"이미지 있는 추천 샘플 {num_samples}개 생성 중...")
    raw_samples = generate_image_recommendation_samples(num_samples)
    print(f"생성 완료: {len(raw_samples)}개")
    
    # 3. 학습 데이터 형식으로 변환 (이미지 랜덤 매칭)
    training_data = convert_to_training_format(raw_samples, images)
    
    # 4. JSONL 저장
    save_to_jsonl(training_data, output_file)
    
    return training_data


if __name__ == "__main__":
    data = get_data(
        image_folder="images/normal_faces",  # 정상 얼굴 이미지 폴더
        num_samples=50,
        output_file="samples/image_recommendation.jsonl"
    )
    
    # 확인용 출력
    print("\n=== 샘플 미리보기 ===")
    for i, sample in enumerate(data[:3]):
        user_content = sample['messages'][1]['content']
        text = user_content[0]['text']
        img_preview = user_content[1]['image_url']['url'][:50] + "..."
        
        tool_call = sample['messages'][2]['tool_calls'][0]
        args = tool_call['function']['arguments']
        
        print(f"\n[{i+1}] Text: {text}")
        print(f"    Image: {img_preview}")
        print(f"    Tool: {tool_call['function']['name']}")
        print(f"    Args: {args}")


## 폴더 구조
"""
project/
├── images/
│   └── normal_faces/       # 정상 얼굴 이미지 (1명)
│       ├── face1.jpg
│       ├── face2.jpg
│       └── ...
├── samples/
│   └── image_recommendation.jsonl
└── generate_image_recommendation.py
```

## 실행 결과 예시
```
이미지 폴더 로드 중: images/normal_faces
로드된 이미지: 5개
이미지 있는 추천 샘플 50개 생성 중...
생성 완료: 50개
저장 완료: samples/image_recommendation.jsonl (50개 샘플)

=== 샘플 미리보기 ===

[1] Text: 이 사진으로 어울리는 헤어스타일 추천해줘~
    Image: data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQAB...
    Tool: hairstyle_recommendation_tool
    Args: {"keywords": ""}

[2] Text: 여름인데 시원하고 가벼운 느낌으로 추천해줘!
    Image: data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQAB...
    Tool: hairstyle_recommendation_tool
    Args: {"keywords": "시원한, 가벼운", "season": "여름"}

[3] Text: 내 얼굴에 맞는 볼륨있는 머리 추천해주세요
    Image: data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQAB...
    Tool: hairstyle_recommendation_tool
    Args: {"keywords": "볼륨있는"}
"""