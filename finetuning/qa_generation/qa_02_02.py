import json
import base64
import random
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

HAIR_LENGTH_CATEGORIES = {
    "male": {
        "숏": "귀 위~귀 아래 길이. 목이 대부분 드러나는 매우 짧은 머리.",
        "미디엄": "귀 아래~턱선 정도 길이. 윗머리 볼륨·가르마 스타일 연출 가능한 중간 기장.",
        "장발": "턱선 아래~어깨 이상 길이. 자연스러운 웨이브 표현 가능한 긴머리."
    },
    "female": {
        "숏": "귀 위~귀 아래 짧은 길이.",
        "단발": "턱선~턱 아래 길이.",
        "중단발": "어깨 위~어깨 닿는 길이. 머리끝이 어깨를 스치는 기장.",
        "미디엄": "어깨 아래~쇄골 길이. 세미 롱으로 자연스러운 웨이브·레이어드 가능.",
        "장발": "쇄골 아래~가슴선 이상 길이. 긴머리 전반을 포함."
    }
}

def generate_image_recommendation_samples(num_samples: int = 50) -> list:
    """이미지 있는 헤어스타일 추천 샘플 생성 (기장 포함)"""
    
    prompt = f"""
헤어스타일 추천 챗봇의 학습 데이터를 생성해주세요.

[시나리오]
사용자가 자신의 사진을 업로드하고 헤어스타일 추천을 요청하는 경우입니다.
이때 hairstyle_recommendation_tool을 호출해야 합니다.

[파라미터 추출 규칙]
1. hairstyle_keywords: 사용자가 원하는 헤어스타일 관련 키워드 (시원한, 가벼운, 볼륨있는, 청순한, 층 내는, 자연스러운, 깔끔한, 댄디한, 펑크한, 귀여운, 시크한 등)
   - 키워드가 없으면 null
   
2. haircolor_keywords: 사용자가 원하는 헤어컬러 관련 키워드 (밝은, 어두운, 톤다운, 브라운, 애쉬, 골드, 레드, 블랙, 자연스러운 색, 적당한 밝기 등)
   - 키워드가 없으면 null
   
3. hairlength_keywords: 사용자가 원하는 기장을 아래 카테고리 중 하나로 변환
   - 남자의 경우: "숏", "미디엄", "장발" 중 하나
   - 여자의 경우: "숏", "단발", "중단발", "미디엄", "장발" 중 하나
   - 기장 언급이 없으면 null
   
   [기장 카테고리 설명]
   남자:
   - 숏: 귀 위~귀 아래 길이. 목이 대부분 드러나는 매우 짧은 머리.
   - 미디엄: 귀 아래~턱선 정도 길이. 윗머리 볼륨·가르마 스타일 연출 가능한 중간 기장.
   - 장발: 턱선 아래~어깨 이상 길이. 자연스러운 웨이브 표현 가능한 긴머리.
   
   여자:
   - 숏: 귀 위~귀 아래 짧은 길이.
   - 단발: 턱선~턱 아래 길이.
   - 중단발: 어깨 위~어깨 닿는 길이. 머리끝이 어깨를 스치는 기장.
   - 미디엄: 어깨 아래~쇄골 길이. 세미 롱으로 자연스러운 웨이브·레이어드 가능.
   - 장발: 쇄골 아래~가슴선 이상 길이. 긴머리 전반을 포함.

4. gender: 질의에서 성별이 명시되거나 유추 가능한 경우 "male" 또는 "female" (없으면 null)

[생성할 질문 유형 - 다양하게 조합]
1. 단순 추천: "이 사진으로 어울리는 머리 추천해줘"
2. 스타일 키워드만: "가벼운 느낌의 머리 추천해줘"
3. 컬러 키워드만: "밝은 색으로 염색하고 싶은데 추천해줘"
4. 기장 키워드만: "짧은 머리 하고 싶어", "어깨 정도 오는 머리 추천해줘"
5. 스타일 + 컬러: "볼륨있고 톤다운된 색으로 추천해줘"
6. 스타일 + 기장: "가벼운 느낌에 단발로 잘라볼까 하는데"
7. 컬러 + 기장: "밝은 색에 긴머리 추천해줘"
8. 스타일 + 컬러 + 기장: "시원한 느낌에 밝은 색, 어깨 정도 오는 기장으로"
9. 계절 포함: "여름이라 시원하고 짧은 머리 추천해줘"
10. 성별 명시: "남자인데 깔끔한 숏컷 추천해줘", "여자인데 단발 어울릴까요?"

[기장 표현 예시 - 이런 표현들을 적절한 카테고리로 변환]
- "짧은 머리", "숏컷", "투블럭" → 숏
- "턱선 정도", "귀 밑으로" → 남자는 미디엄
- "단발", "턱 아래 정도" → 단발
- "어깨 정도", "쇄골 위", "적당한 길이" → 중단발
- "쇄골 아래", "세미롱" → 미디엄
- "긴머리", "장발", "허리까지" → 장발

다양한 말투로 {num_samples}개 생성해주세요.
- "이 사진", "내 사진", "이 얼굴", "내 얼굴" 등 다양한 표현 사용
- 반말, 존댓말, 이모지 등 다양하게
- 모든 파라미터가 있는 경우, 일부만 있는 경우, 아무것도 없는 경우 골고루 포함

[출력 형식]
JSON 배열로 출력. 각 항목:
{{
  "user": "사용자 질의 (이미지 언급 포함)",
  "hairstyle_keywords": "키워드1, 키워드2" 또는 null,
  "haircolor_keywords": "키워드1, 키워드2" 또는 null,
  "hairlength_keywords": "카테고리명" 또는 null,
  "gender": "male" 또는 "female" 또는 null
}}

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


def convert_to_training_format(samples: list) -> list:
    """생성된 샘플을 학습 데이터 형식으로 변환 (이미지는 build_dataset.py에서 매칭)"""

    training_data = []

    for i, sample in enumerate(samples):

        arguments = {}

        if sample.get("hairstyle_keywords"):
            arguments["hairstyle_keywords"] = sample["hairstyle_keywords"]

        if sample.get("haircolor_keywords"):
            arguments["haircolor_keywords"] = sample["haircolor_keywords"]

        if sample.get("hairlength_keywords"):
            arguments["hairlength_keywords"] = sample["hairlength_keywords"]

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
                                "name": "hairstyle_recommendation_tool",
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
    
    with open(filename, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"저장 완료: {filename} ({len(data)}개 샘플)")


def analyze_samples(samples: list):
    """생성된 샘플의 파라미터 분포 분석"""
    stats = {
        "total": len(samples),
        "has_hairstyle": 0,
        "has_haircolor": 0,
        "has_hairlength": 0,
        "has_gender": 0,
        "hairlength_distribution": {},
        "gender_distribution": {"male": 0, "female": 0, "null": 0}
    }
    
    for sample in samples:
        if sample.get("hairstyle_keywords"):
            stats["has_hairstyle"] += 1
        if sample.get("haircolor_keywords"):
            stats["has_haircolor"] += 1
        if sample.get("hairlength_keywords"):
            stats["has_hairlength"] += 1
            length = sample["hairlength_keywords"]
            stats["hairlength_distribution"][length] = stats["hairlength_distribution"].get(length, 0) + 1
        if sample.get("gender"):
            stats["has_gender"] += 1
            stats["gender_distribution"][sample["gender"]] += 1
        else:
            stats["gender_distribution"]["null"] += 1
    
    print("\n=== 샘플 분포 분석 ===")
    print(f"총 샘플 수: {stats['total']}")
    print(f"hairstyle_keywords 포함: {stats['has_hairstyle']}개 ({stats['has_hairstyle']/stats['total']*100:.1f}%)")
    print(f"haircolor_keywords 포함: {stats['has_haircolor']}개 ({stats['has_haircolor']/stats['total']*100:.1f}%)")
    print(f"hairlength_keywords 포함: {stats['has_hairlength']}개 ({stats['has_hairlength']/stats['total']*100:.1f}%)")
    print(f"gender 포함: {stats['has_gender']}개 ({stats['has_gender']/stats['total']*100:.1f}%)")
    print(f"\n기장 분포: {stats['hairlength_distribution']}")
    print(f"성별 분포: {stats['gender_distribution']}")
    
    return stats


def get_data(
    num_samples: int = 10,
    output_file: str = "finetuning/samples/qa_02_02.jsonl"
):
    """메인 함수: 텍스트 질의-응답 생성 (이미지는 build_dataset.py에서 매칭)"""

    print(f"이미지 있는 추천 샘플 {num_samples}개 생성 중...")
    raw_samples = generate_image_recommendation_samples(num_samples)
    print(f"생성 완료: {len(raw_samples)}개")

    analyze_samples(raw_samples)

    training_data = convert_to_training_format(raw_samples)

    save_to_jsonl(training_data, output_file)

    return training_data


if __name__ == "__main__":
    data = get_data(
        num_samples=100,
        output_file="finetuning/samples/qa_02_02.jsonl"
    )
    
    print("\n=== 샘플 미리보기 ===")
    for i, sample in enumerate(data[:5]):
        user_content = sample['messages'][1]['content']
        text = user_content[0]['text']
        img_preview = user_content[1]['image_url']['url'][:50] + "..."
        
        tool_call = sample['messages'][2]['tool_calls'][0]
        args = tool_call['function']['arguments']
        
        print(f"\n[{i+1}] Text: {text}")
        print(f"    Image: {img_preview}")
        print(f"    Tool: {tool_call['function']['name']}")
        print(f"    Args: {args}")