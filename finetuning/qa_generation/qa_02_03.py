import json
import base64
import random
from pathlib import Path
from openai import OpenAI

client = OpenAI()

# 지원되는 옵션 목록
HAIRSTYLES = {
    "male_cut": ["가일컷", "댄디컷", "드랍컷", "리젠트컷", "리프컷", "스왓컷", "아이비리그컷", "울프컷", "크롭컷", "포마드컷", "짧은포마드컷", "필러스컷"],
    "male_perm": ["가르마펌", "가일펌", "댄디펌", "리젠트펌", "리프펌", "베이비펌", "볼륨펌", "쉐도우펌", "스왈로펌", "애즈펌", "울프펌", "크리드펌", "포마드펌", "히피펌"],
    "female_cut": ["레이어드컷", "리프컷", "머쉬룸컷", "뱅헤어", "보브컷", "샤기컷", "원랭스컷", "픽시컷", "허쉬컷", "히메컷"],
    "female_perm": ["C컬펌", "S컬펌", "글램펌", "내츄럴펌", "러블리펌", "루즈펌", "리프펌", "물결펌", "바디펌", "발롱펌", "볼드펌", "볼륨매직", "볼륨펌", "빌드펌", "에어펌", "젤리펌", "지젤펌", "쿠션펌", "텍스처펌", "퍼피베이비펌", "허쉬펌_롱"]
}

HAIRCOLORS = [
    "골드브라운", "다크브라운", "레드브라운", "레드와인", "로즈골드", "마르살라", "마호가니",
    "밀크브라운", "베이지브라운", "블루블랙", "애쉬그레이", "애쉬바이올렛", "애쉬베이지",
    "애쉬브라운", "애쉬블론드", "애쉬블루", "애쉬카키", "애쉬퍼플",
    "오렌지브라운", "올리브브라운", "초코브라운", "카키브라운", "쿠퍼브라운", "핑크브라운"
]

ALL_HAIRSTYLES = HAIRSTYLES["male_cut"] + HAIRSTYLES["male_perm"] + HAIRSTYLES["female_cut"] + HAIRSTYLES["female_perm"]


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
    
    print(f"로드된 이미지: {len(images)}개")
    return images


def generate_image_generation_samples(num_samples: int = 50) -> list:
    """이미지 생성 도구 호출 샘플 생성"""
    
    hairstyles_str = ", ".join(ALL_HAIRSTYLES)
    haircolors_str = ", ".join(HAIRCOLORS)
    
    prompt = f"""
헤어스타일 추천 챗봇의 학습 데이터를 생성해주세요.

[시나리오]
사용자가 자신의 사진에 특정 헤어스타일이나 헤어컬러를 적용해달라고 요청하는 경우입니다.
이때 hairstyle_generation_tool을 호출해야 합니다.

[중요] 반드시 아래 목록에 있는 스타일/컬러만 사용하세요!

[지원되는 헤어스타일]
{hairstyles_str}

[지원되는 헤어컬러]
{haircolors_str}

[생성할 질문 유형 - 다양한 조합]
1. 헤어스타일만: "히피펌으로 바꿔줘"
2. 헤어컬러만: "애쉬그레이로 염색해줘"
3. 둘 다: "C컬펌이랑 밀크브라운으로 해줘"

[다양한 표현 사용]
- "적용해줘", "바꿔줘", "해줘", "변경해줘", "합성해줘"
- "~로 염색해줘", "~색으로 바꿔줘"
- "이 사진에", "내 사진에", "이 얼굴에"
- 오타/띄어쓰기 변형도 포함: "리젠트 펌", "애쉬 그레이", "C컬 펌" 등

[출력 규칙]
- hairstyle: 반드시 위 목록에 있는 정확한 이름으로 정규화 (띄어쓰기 제거)
- haircolor: 반드시 위 목록에 있는 정확한 이름으로 정규화

{num_samples}개 생성해주세요. 다양한 말투(반말, 존댓말, 이모지)로 만들어주세요.

[출력 형식]
JSON 배열로 출력. 각 항목:
{{
  "user": "사용자 질의 (이미지 언급 포함)",
  "hairstyle": "정규화된 헤어스타일명" (없으면 null),
  "haircolor": "정규화된 헤어컬러명" (없으면 null)
}}

최소 하나(hairstyle 또는 haircolor)는 반드시 있어야 합니다.
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


def validate_and_filter_samples(samples: list) -> list:
    """생성된 샘플 검증 - 지원되는 옵션만 필터링"""
    
    valid_samples = []
    
    for sample in samples:
        hairstyle = sample.get("hairstyle")
        haircolor = sample.get("haircolor")
        
        # 최소 하나는 있어야 함
        if not hairstyle and not haircolor:
            continue
        
        # 유효성 검사
        if hairstyle and hairstyle not in ALL_HAIRSTYLES:
            print(f"[WARNING] 지원되지 않는 헤어스타일 제외: {hairstyle}")
            continue
        
        if haircolor and haircolor not in HAIRCOLORS:
            print(f"[WARNING] 지원되지 않는 헤어컬러 제외: {haircolor}")
            continue
        
        valid_samples.append(sample)
    
    print(f"유효한 샘플: {len(valid_samples)}/{len(samples)}개")
    return valid_samples


def convert_to_training_format(samples: list, images: list) -> list:
    """생성된 샘플을 학습 데이터 형식으로 변환"""
    
    training_data = []
    
    for i, sample in enumerate(samples):
        # 랜덤 이미지 선택
        selected_image = random.choice(images)
        
        # arguments 구성
        arguments = {}
        if sample.get("hairstyle"):
            arguments["hairstyle"] = sample["hairstyle"]
        if sample.get("haircolor"):
            arguments["haircolor"] = sample["haircolor"]
        
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
                                "name": "hairstyle_generation_tool",
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
    output_file: str = "image_generation.jsonl"
):
    """메인 함수: 이미지 로드 → 데이터 생성 → 검증 → 변환 → 저장"""
    
    # 1. 이미지 로드
    print(f"이미지 폴더 로드 중: {image_folder}")
    images = load_images_as_base64(image_folder)
    
    if not images:
        raise ValueError(f"이미지가 없습니다: {image_folder}")
    
    # 2. GPT로 샘플 생성
    print(f"이미지 생성 샘플 {num_samples}개 생성 중...")
    raw_samples = generate_image_generation_samples(num_samples)
    print(f"생성 완료: {len(raw_samples)}개")
    
    # 3. 유효성 검증
    valid_samples = validate_and_filter_samples(raw_samples)
    
    # 4. 학습 데이터 형식으로 변환
    training_data = convert_to_training_format(valid_samples, images)
    
    # 5. JSONL 저장
    save_to_jsonl(training_data, output_file)
    
    return training_data


if __name__ == "__main__":
    data = get_data(
        image_folder="images/normal_faces",
        num_samples=60,  # 검증에서 일부 제외될 수 있으니 여유있게
        output_file="samples/image_generation.jsonl"
    )
    
    # 확인용 출력
    print("\n=== 샘플 미리보기 ===")
    for i, sample in enumerate(data[:5]):
        user_content = sample['messages'][1]['content']
        text = user_content[0]['text']
        
        tool_call = sample['messages'][2]['tool_calls'][0]
        args = tool_call['function']['arguments']
        
        print(f"\n[{i+1}] User: {text}")
        print(f"    Args: {args}")

## 실행 결과 예시
"""
이미지 폴더 로드 중: images/normal_faces
로드된 이미지: 5개
이미지 생성 샘플 60개 생성 중...
생성 완료: 60개
[WARNING] 지원되지 않는 헤어스타일 제외: 투블럭
유효한 샘플: 58/60개
저장 완료: samples/image_generation.jsonl (58개 샘플)

=== 샘플 미리보기 ===

[1] User: 이 사진에 히피펌 적용해줘~
    Args: {"hairstyle": "히피펌"}

[2] User: 애쉬그레이로 염색한 모습 보여줘
    Args: {"haircolor": "애쉬그레이"}

[3] User: C컬펌이랑 밀크브라운으로 바꿔줘!
    Args: {"hairstyle": "C컬펌", "haircolor": "밀크브라운"}

[4] User: 내 사진에 리젠트펌 해줘
    Args: {"hairstyle": "리젠트펌"}

[5] User: 레이어드컷에 로즈골드 색으로 변경해주세요
    Args: {"hairstyle": "레이어드컷", "haircolor": "로즈골드"}
"""