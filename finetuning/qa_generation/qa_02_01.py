import json
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

def generate_no_image_recommendation_samples(num_samples: int = 10) -> list:
    """이미지 없는 헤어스타일 추천 샘플 생성"""
    
    prompt = f"""
헤어스타일 추천 챗봇의 학습 데이터를 생성해주세요.

[시나리오]
사용자가 이미지 없이 헤어스타일 추천을 요청하는 경우입니다.
이때 non_image_recommendation_tool을 호출해야 합니다.

[필수 조건]
도구 호출이 가능하려면 다음 중 하나를 만족해야 함:
- 성별 + 얼굴형이 모두 있음
- 퍼스널컬러가 있음

[파라미터 추출 규칙]
1. gender: 여자→"Female", 남자→"Male"
2. face_shape: 둥근형→"Round", 사각형→"Square", 하트형→"Heart", 계란형→"Oval", 긴형→"Oblong"
3. personal_color: "봄 웜톤", "가을 웜톤", "겨울 쿨톤", "여름 쿨톤" 중 하나
4. season: 봄, 여름, 가을, 겨울 (언급된 경우만)
5. hairstyle_keywords: 원하는 스타일 키워드 (가벼운, 시원한, 볼륨있는, 청순한 등) - 컬러 제외
6. haircolor_keywords: 원하는 컬러 키워드 (톤다운, 밝은, 어두운, 화사한 등)
7. hairlength_keywords: 원하는 기장 (성별에 따라 다름)
   - 남자: "숏", "미디엄", "장발" 중 하나
     * 숏: 귀 위~귀 아래 길이. 목이 대부분 드러나는 매우 짧은 머리.
     * 미디엄: 귀 아래~턱선 정도 길이. 윗머리 볼륨·가르마 스타일 연출 가능한 중간 기장.
     * 장발: 턱선 아래~어깨 이상 길이. 자연스러운 웨이브 표현 가능한 긴머리.
   - 여자: "숏", "단발", "중단발", "미디엄", "장발" 중 하나
     * 숏: 귀 위~귀 아래 짧은 길이.
     * 단발: 턱선~턱 아래 길이
     * 중단발: 어깨 위~어깨 닿는 길이. 머리끝이 어깨를 스치는 기장.
     * 미디엄: 어깨 아래~쇄골 길이. 세미 롱으로 자연스러운 웨이브·레이어드 가능.
     * 장발: 쇄골 아래~가슴선 이상 길이. 긴머리 전반을 포함.

[기장 추출 예시]
- "짧은 머리 하고 싶어" (여자) → "숏" 또는 "단발"
- "어깨 정도 길이로" (여자) → "중단발"
- "긴 머리 기르고 싶어" → "장발"
- "너무 짧지 않고 적당한 길이" (여자) → "중단발" 또는 "미디엄"
- "목이 시원하게 보이는 길이" (남자) → "숏"
- "가르마 스타일 하고 싶어" (남자) → "미디엄"

[생성 규칙]
다양한 조합으로 {num_samples}개 생성:
- 성별 + 얼굴형만 있는 경우
- 성별 + 얼굴형 + 계절이 있는 경우
- 성별 + 얼굴형 + 스타일 키워드가 있는 경우
- 성별 + 얼굴형 + 컬러 키워드가 있는 경우
- 성별 + 얼굴형 + 기장 + 스타일 키워드가 있는 경우
- 퍼스널컬러만 있는 경우
- 퍼스널컬러 + 기장이 있는 경우
- 퍼스널컬러 + 키워드가 있는 경우
- 모든 정보가 다 있는 경우

기장 관련 표현 다양하게:
- 직접적: "숏컷", "단발", "장발", "짧은 머리", "긴 머리"
- 간접적: "시원하게", "어깨까지", "쇄골 정도", "턱선 길이", "귀 아래로", "목이 보이게"
- 상대적: "지금보다 짧게", "너무 짧지 않게", "적당한 길이로"

다양한 말투와 표현을 사용해주세요 (반말, 존댓말, 이모지 등)

[출력 형식]
JSON 배열로 출력. 각 항목:
{{
  "user": "사용자 질의",
  "arguments": {{
    "gender": "Female" 또는 "Male" (있는 경우만),
    "face_shape": "Round" 등 (있는 경우만),
    "personal_color": "봄 웜톤" 등 (있는 경우만),
    "season": "여름" 등 (있는 경우만),
    "hairstyle_keywords": "가벼운, 시원한" (있는 경우만),
    "haircolor_keywords": "톤다운" (있는 경우만),
    "hairlength_keywords": "중단발" (있는 경우만)
  }}
}}

arguments에는 질의에서 추출 가능한 파라미터만 포함하세요.
hairlength_keywords는 반드시 정해진 카테고리 중 하나로 변환하세요.
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


def validate_hairlength(gender: str, hairlength: str) -> str | None:
    """기장이 성별에 맞는 유효한 값인지 검증하고 수정"""
    male_lengths = ["숏", "미디엄", "장발"]
    female_lengths = ["숏", "단발", "중단발", "미디엄", "장발"]
    
    if gender == "Male":
        if hairlength in male_lengths:
            return hairlength
        if hairlength == "단발":
            return "미디엄"
        if hairlength == "중단발":
            return "미디엄"
        return None
    elif gender == "Female":
        if hairlength in female_lengths:
            return hairlength
        return None
    return hairlength


def convert_to_training_format(samples: list) -> list:
    """생성된 샘플을 학습 데이터 형식으로 변환"""
    
    training_data = []
    
    for i, sample in enumerate(samples):
        arguments = sample["arguments"]
        
        if "hairlength_keywords" in arguments and "gender" in arguments:
            validated_length = validate_hairlength(
                arguments["gender"], 
                arguments["hairlength_keywords"]
            )
            if validated_length:
                arguments["hairlength_keywords"] = validated_length
            else:
                del arguments["hairlength_keywords"]

        arguments_str = json.dumps(arguments, ensure_ascii=False)        

        training_sample = {
            "messages": [
                {"role": "system", "content": ""},
                {"role": "user", "content": sample["user"]},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call_{i+1:03d}",
                            "type": "function",
                            "function": {
                                "name": "non_image_recommendation_tool",
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
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(filename, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"저장 완료: {filename} ({len(data)}개 샘플)")


def get_data(num_samples: int = 50, output_file: str = "finetuning/samples/qa_02_01.jsonl"):
    """메인 함수: 데이터 생성 → 변환 → 저장"""
    
    print(f"이미지 없는 추천 샘플 {num_samples}개 생성 중...")
    
    raw_samples = generate_no_image_recommendation_samples(num_samples)
    print(f"생성 완료: {len(raw_samples)}개")
    
    training_data = convert_to_training_format(raw_samples)
    
    save_to_jsonl(training_data, output_file)
    
    return training_data

def print_statistics(data: list):
    """생성된 데이터 통계 출력"""
    stats = {
        "total": len(data),
        "with_gender": 0,
        "with_face_shape": 0,
        "with_personal_color": 0,
        "with_season": 0,
        "with_hairstyle_keywords": 0,
        "with_haircolor_keywords": 0,
        "with_hairlength_keywords": 0,
    }
    
    hairlength_dist = {}
    
    for sample in data:
        args = json.loads(sample['messages'][2]['tool_calls'][0]['function']['arguments'])
        
        if "gender" in args:
            stats["with_gender"] += 1
        if "face_shape" in args:
            stats["with_face_shape"] += 1
        if "personal_color" in args:
            stats["with_personal_color"] += 1
        if "season" in args:
            stats["with_season"] += 1
        if "hairstyle_keywords" in args:
            stats["with_hairstyle_keywords"] += 1
        if "haircolor_keywords" in args:
            stats["with_haircolor_keywords"] += 1
        if "hairlength_keywords" in args:
            stats["with_hairlength_keywords"] += 1
            length = args["hairlength_keywords"]
            hairlength_dist[length] = hairlength_dist.get(length, 0) + 1
    
    print("\n=== 데이터 통계 ===")
    print(f"총 샘플 수: {stats['total']}")
    print(f"성별 포함: {stats['with_gender']}")
    print(f"얼굴형 포함: {stats['with_face_shape']}")
    print(f"퍼스널컬러 포함: {stats['with_personal_color']}")
    print(f"계절 포함: {stats['with_season']}")
    print(f"스타일 키워드 포함: {stats['with_hairstyle_keywords']}")
    print(f"컬러 키워드 포함: {stats['with_haircolor_keywords']}")
    print(f"기장 키워드 포함: {stats['with_hairlength_keywords']}")
    
    if hairlength_dist:
        print("\n=== 기장 분포 ===")
        for length, count in sorted(hairlength_dist.items(), key=lambda x: -x[1]):
            print(f"  {length}: {count}")


if __name__ == "__main__":
    data = get_data(num_samples=100, output_file="finetuning/samples/qa_02_01.jsonl")

    print_statistics(data)

    print("\n=== 샘플 미리보기 ===")
    for i, sample in enumerate(data[:3]):
        user_msg = sample['messages'][1]['content']
        tool_call = sample['messages'][2]['tool_calls'][0]
        args = tool_call['function']['arguments']

        print(f"\n[{i+1}] User: {user_msg}")
        print(f"    Tool: {tool_call['function']['name']}")
        print(f"    Args: {args}")


"""
이미지 없는 추천 샘플 50개 생성 중...
생성 완료: 50개
저장 완료: samples/no_image_recommendation.jsonl (50개 샘플)

=== 샘플 미리보기 ===

[1] User: 나 여자고 둥근 얼굴이야~ 가벼운 머리 추천해줘!
    Tool: non_image_recommendation_tool
    Args: {"gender": "Female", "face_shape": "Round", "hairstyle_keywords": "가벼운"}

[2] User: 봄 웜톤인데 어울리는 염색 색 추천해주세요
    Tool: non_image_recommendation_tool
    Args: {"personal_color": "봄 웜톤"}

[3] User: 남자고 사각턱인데 여름이라 시원하고 짧은 머리 하고싶어. 밝은 색으로 염색도 할까 생각중이야
    Tool: non_image_recommendation_tool
    Args: {"gender": "Male", "face_shape": "Square", "season": "여름", "hairstyle_keywords": "시원한, 짧은", "haircolor_keywords": "밝은"}

"""