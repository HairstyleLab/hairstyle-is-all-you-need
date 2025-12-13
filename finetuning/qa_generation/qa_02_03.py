import json
import base64
import random
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

HAIRSTYLES = {
    "male_cut": ["가일컷", "댄디컷", "드랍컷", "리젠트컷", "리프컷", "버즈컷", "슬릭백컷", "아이비리그컷", "울프컷", "크롭컷", "크루컷", "투블럭컷", "페이드컷", "포마드컷", "필러스컷", "하이앤타이트컷", "가르마펌"],
    "male_perm": ["가일펌", "내추럴펌", "댄디펌", "리젠트펌", "리프펌", "베이비펌", "볼륨펌", "쉐도우펌", "스왈로펌", "애즈펌", "웨이브펌", "크리드펌", "포마드펌", "히피펌"],
    "female_cut": ["레이어드컷", "리프컷", "머쉬룸컷", "뱅헤어", "보브컷", "샤기컷", "원랭스컷", "픽시컷", "허쉬컷", "히메컷"],
    "female_perm": ["CS컬펌", "C컬펌", "S컬펌", "글램펌", "내츄럴펌", "디지털펌", "러블리펌", "레이어드펌", "루즈펌", "리프펌", "물결펌", "믹스펌", "바디펌", "발롱펌", "볼드펌", "볼륨매직", "볼륨펌", "빌드펌", "셋팅펌", "스파이럴펌", "에어펌", "젤리펌", "지젤펌", "쿠션펌", "텍스처펌", "퍼피베이비펌", "허쉬펌", "히피펌"]
}

HAIRCOLORS = [
    "골드브라운", "다크브라운", "레드브라운", "레드와인", "로즈골드", "마르살라", "마호가니",
    "밀크브라운", "베이지브라운", "블루블랙", "애쉬그레이", "애쉬바이올렛", "애쉬베이지",
    "애쉬브라운", "애쉬블론드", "애쉬블루", "애쉬카키", "애쉬퍼플",
    "오렌지브라운", "올리브브라운", "초코브라운", "카키브라운", "쿠퍼브라운", "핑크브라운"
]

HAIRLENGTHS = {
    "male": ["숏", "미디엄", "장발"],
    "female": ["숏", "단발", "중단발", "미디엄", "장발"]
}

HAIRLENGTH_DESCRIPTIONS = {
    "male": {
        "숏": "귀 위~귀 아래 길이. 목이 대부분 드러나는 매우 짧은 머리",
        "미디엄": "귀 아래~턱선 정도 길이. 윗머리 볼륨·가르마 스타일 연출 가능한 중간 기장",
        "장발": "턱선 아래~어깨 이상 길이. 자연스러운 웨이브 표현 가능한 긴머리"
    },
    "female": {
        "숏": "귀 위~귀 아래 짧은 길이",
        "단발": "턱선~턱 아래 길이",
        "중단발": "어깨 위~어깨 닿는 길이. 머리끝이 어깨를 스치는 기장",
        "미디엄": "어깨 아래~쇄골 길이. 세미 롱으로 자연스러운 웨이브·레이어드 가능",
        "장발": "쇄골 아래~가슴선 이상 길이. 긴머리 전반을 포함"
    }
}

ALL_HAIRSTYLES = HAIRSTYLES["male_cut"] + HAIRSTYLES["male_perm"] + HAIRSTYLES["female_cut"] + HAIRSTYLES["female_perm"]
ALL_HAIRLENGTHS = list(set(HAIRLENGTHS["male"] + HAIRLENGTHS["female"]))


def load_preprocessed_images(json_file: str) -> list:
    """미리 인코딩된 base64 이미지 JSON 파일 로드"""
    with open(json_file, "r", encoding="utf-8") as f:
        images = json.load(f)
    
    print(f"로드된 이미지: {len(images)}개")
    return images


def generate_image_generation_samples(num_samples: int = 50) -> list:
    """이미지 생성 도구 호출 샘플 생성 (기장 포함)"""
    
    hairstyles_str = ", ".join(ALL_HAIRSTYLES)
    haircolors_str = ", ".join(HAIRCOLORS)
    
    prompt = f"""
헤어스타일 추천 챗봇의 학습 데이터를 생성해주세요.

[시나리오]
사용자가 자신의 사진에 특정 헤어스타일, 헤어컬러, 또는 기장을 적용해달라고 요청하는 경우입니다.
이때 hairstyle_generation_tool을 호출해야 합니다.

[중요] 반드시 아래 목록에 있는 스타일/컬러/기장만 사용하세요!

[지원되는 헤어스타일]
{hairstyles_str}

[지원되는 헤어컬러]
{haircolors_str}

[지원되는 기장 옵션]
- 남자: 숏, 미디엄, 장발
- 여자: 숏, 단발, 중단발, 미디엄, 장발

[기장 설명]
남자:
- 숏: 귀 위~귀 아래 길이. 목이 대부분 드러나는 매우 짧은 머리
- 미디엄: 귀 아래~턱선 정도 길이. 윗머리 볼륨·가르마 스타일 연출 가능한 중간 기장
- 장발: 턱선 아래~어깨 이상 길이. 자연스러운 웨이브 표현 가능한 긴머리

여자:
- 숏: 귀 위~귀 아래 짧은 길이
- 단발: 턱선~턱 아래 길이
- 중단발: 어깨 위~어깨 닿는 길이. 머리끝이 어깨를 스치는 기장
- 미디엄: 어깨 아래~쇄골 길이. 세미 롱으로 자연스러운 웨이브·레이어드 가능
- 장발: 쇄골 아래~가슴선 이상 길이. 긴머리 전반을 포함

[생성할 질문 유형 - 다양한 조합으로 균등 분배]
1. 헤어스타일만: "히피펌으로 바꿔줘" (약 15%)
2. 헤어컬러만: "애쉬그레이로 염색해줘" (약 15%)
3. 기장만: "단발로 잘라줘", "숏컷으로 해줘" (약 10%)
4. 스타일 + 컬러: "C컬펌이랑 밀크브라운으로 해줘" (약 15%)
5. 스타일 + 기장: "히피펌인데 장발로 해줘", "숏으로 애즈펌 해줘" (약 15%)
6. 컬러 + 기장: "애쉬그레이에 단발로 해줘" (약 15%)
7. 스타일 + 컬러 + 기장: "중단발 C컬펌에 로즈골드로 해줘" (약 15%)

[기장 표현 예시 - 다양하게 활용]
- 직접 표현: "숏으로", "단발로", "장발로", "중단발로"
- 길이 묘사: "짧게", "길게", "어깨까지", "쇄골까지", "턱선까지", "귀 아래로"
- 구체적 묘사: "귀가 보이게", "목이 드러나게", "어깨에 닿게", "쇄골 정도로"
- 변경 요청: "좀 더 길게", "짧게 잘라서", "기장은 미디엄으로"

[다양한 표현 사용]
- "적용해줘", "바꿔줘", "해줘", "변경해줘", "합성해줘", "만들어줘"
- "~로 염색해줘", "~색으로 바꿔줘"
- "이 사진에", "내 사진에", "이 얼굴에", "여기에"
- 오타/띄어쓰기 변형도 포함: "리젠트 펌", "애쉬 그레이", "C컬 펌", "중 단발" 등

[출력 규칙]
- hairstyle: 반드시 위 목록에 있는 정확한 이름으로 정규화 (띄어쓰기 제거)
- haircolor: 반드시 위 목록에 있는 정확한 이름으로 정규화
- hairlength: 사용자의 기장 묘사를 분석해 적절한 카테고리로 매핑
  - 여성 스타일이면: 숏, 단발, 중단발, 미디엄, 장발 중 선택
  - 남성 스타일이면: 숏, 미디엄, 장발 중 선택
  - 성별 불명확하면: 문맥에 맞게 선택

[기장 매핑 예시]
- "짧게 해줘", "귀 위로", "목 보이게" → 숏
- "턱선까지", "턱 아래로" (여자) → 단발
- "어깨까지", "어깨에 닿게" → 중단발
- "쇄골까지", "어깨 아래로" → 미디엄
- "길게", "가슴까지", "허리까지" → 장발

{num_samples}개 생성해주세요. 다양한 말투(반말, 존댓말, 이모지)로 만들어주세요.

[출력 형식]
JSON 배열로 출력. 각 항목:
{{
  "user": "사용자 질의 (이미지 언급 포함)",
  "hairstyle": "정규화된 헤어스타일명" (없으면 null),
  "haircolor": "정규화된 헤어컬러명" (없으면 null),
  "hairlength": "정규화된 기장" (없으면 null)
}}

최소 하나(hairstyle, haircolor, hairlength 중)는 반드시 있어야 합니다.
JSON 배열만 출력하고 다른 설명은 하지 마세요.
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
    
    samples = json.loads(content)
    return samples


def validate_and_filter_samples(samples: list) -> list:
    """생성된 샘플 검증 - 지원되는 옵션만 필터링"""
    
    valid_samples = []
    
    for sample in samples:
        hairstyle = sample.get("hairstyle")
        haircolor = sample.get("haircolor")
        hairlength = sample.get("hairlength")
        
        if not hairstyle and not haircolor and not hairlength:
            print(f"[WARNING] 빈 샘플 제외: {sample}")
            continue
        
        if hairstyle and hairstyle not in ALL_HAIRSTYLES:
            print(f"[WARNING] 지원되지 않는 헤어스타일 제외: {hairstyle}")
            continue
        
        if haircolor and haircolor not in HAIRCOLORS:
            print(f"[WARNING] 지원되지 않는 헤어컬러 제외: {haircolor}")
            continue
        
        if hairlength and hairlength not in ALL_HAIRLENGTHS:
            print(f"[WARNING] 지원되지 않는 기장 제외: {hairlength}")
            continue
        
        valid_samples.append(sample)
    
    print(f"유효한 샘플: {len(valid_samples)}/{len(samples)}개")
    return valid_samples


def convert_to_training_format(samples: list) -> list:
    """생성된 샘플을 학습 데이터 형식으로 변환 (이미지는 build_dataset.py에서 매칭)"""

    training_data = []

    for i, sample in enumerate(samples):

        arguments = {}
        if sample.get("hairstyle"):
            arguments["hairstyle"] = sample["hairstyle"]
        if sample.get("haircolor"):
            arguments["haircolor"] = sample["haircolor"]
        if sample.get("hairlength"):
            arguments["hairlength"] = sample["hairlength"]

        arguments_str = json.dumps(arguments, ensure_ascii=False)

        training_sample = {
            "messages": [
                {"role": "system", "content": ""},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": sample["user"]},
                        {"type": "image_url", "image_url": {"url": ""}}
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
            ],
            "image_type": "normal"
        }
        training_data.append(training_sample)

    return training_data


def save_to_jsonl(data: list, filename: str):
    """JSONL 파일로 저장"""
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"저장 완료: {filename} ({len(data)}개 샘플)")


def analyze_distribution(samples: list):
    """생성된 샘플의 분포 분석"""
    
    stats = {
        "style_only": 0,
        "color_only": 0,
        "length_only": 0,
        "style_color": 0,
        "style_length": 0,
        "color_length": 0,
        "all_three": 0
    }
    
    for sample in samples:
        has_style = bool(sample.get("hairstyle"))
        has_color = bool(sample.get("haircolor"))
        has_length = bool(sample.get("hairlength"))
        
        if has_style and has_color and has_length:
            stats["all_three"] += 1
        elif has_style and has_color:
            stats["style_color"] += 1
        elif has_style and has_length:
            stats["style_length"] += 1
        elif has_color and has_length:
            stats["color_length"] += 1
        elif has_style:
            stats["style_only"] += 1
        elif has_color:
            stats["color_only"] += 1
        elif has_length:
            stats["length_only"] += 1
    
    total = len(samples)
    print("\n=== 샘플 분포 분석 ===")
    for key, count in stats.items():
        pct = (count / total * 100) if total > 0 else 0
        print(f"  {key}: {count}개 ({pct:.1f}%)")
    
    return stats


def get_data(
    num_samples: int = 10,
    output_file: str = "finetuning/samples/qa_02_03.jsonl"
):
    """메인 함수: 텍스트 질의-응답 생성 (이미지는 build_dataset.py에서 매칭)"""

    print(f"이미지 생성 샘플 {num_samples}개 생성 중...")
    raw_samples = generate_image_generation_samples(num_samples)
    print(f"생성 완료: {len(raw_samples)}개")

    valid_samples = validate_and_filter_samples(raw_samples)

    analyze_distribution(valid_samples)

    training_data = convert_to_training_format(valid_samples)

    save_to_jsonl(training_data, output_file)

    return training_data


if __name__ == "__main__":
    data = get_data(
        num_samples=100,
        output_file="finetuning/samples/qa_02_03.jsonl"
    )
    
    print("\n=== 샘플 미리보기 ===")
    for i, sample in enumerate(data[:10]):
        user_content = sample['messages'][1]['content']
        text = user_content[0]['text']
        
        tool_call = sample['messages'][2]['tool_calls'][0]
        args = tool_call['function']['arguments']
        
        print(f"\n[{i+1}] User: {text}")
        print(f"    Args: {args}")


"""
이미지 폴더 로드 중: images/normal_faces
로드된 이미지: 5개
이미지 생성 샘플 80개 생성 중...
생성 완료: 80개
유효한 샘플: 76/80개

=== 샘플 분포 분석 ===
  style_only: 12개 (15.8%)
  color_only: 11개 (14.5%)
  length_only: 8개 (10.5%)
  style_color: 12개 (15.8%)
  style_length: 11개 (14.5%)
  color_length: 10개 (13.2%)
  all_three: 12개 (15.8%)

저장 완료: samples/image_generation_with_length.jsonl (76개 샘플)

=== 샘플 미리보기 ===

[1] User: 이 사진에 히피펌 적용해줘~
    Args: {"hairstyle": "히피펌"}

[2] User: 애쉬그레이로 염색한 모습 보여줘
    Args: {"haircolor": "애쉬그레이"}

[3] User: 단발로 잘라줘!
    Args: {"hairlength": "단발"}

[4] User: C컬펌이랑 밀크브라운으로 바꿔줘!
    Args: {"hairstyle": "C컬펌", "haircolor": "밀크브라운"}

[5] User: 히피펌인데 장발로 해줘
    Args: {"hairstyle": "히피펌", "hairlength": "장발"}

[6] User: 애쉬그레이에 어깨까지 길이로 해줄래?
    Args: {"haircolor": "애쉬그레이", "hairlength": "중단발"}

[7] User: 중단발 C컬펌에 로즈골드로 해줘
    Args: {"hairstyle": "C컬펌", "haircolor": "로즈골드", "hairlength": "중단발"}

[8] User: 숏으로 애즈펌 해주세요
    Args: {"hairstyle": "애즈펌", "hairlength": "숏"}

[9] User: 쇄골까지 오게 레이어드컷 해줘
    Args: {"hairstyle": "레이어드컷", "hairlength": "미디엄"}

[10] User: 귀 아래로 짧게 자르고 블루블랙으로 염색해줘
    Args: {"haircolor": "블루블랙", "hairlength": "숏"}
"""