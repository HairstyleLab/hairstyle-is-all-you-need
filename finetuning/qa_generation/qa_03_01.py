import json
from openai import OpenAI

client = OpenAI()

def generate_no_image_exception_samples(num_samples: int = 50) -> list:
    """이미지 없는 추천 예외처리 샘플 생성"""
    
    prompt = f"""
헤어스타일 추천 챗봇의 학습 데이터를 생성해주세요.

[시나리오]
사용자가 이미지 없이 헤어스타일 추천을 요청하지만, 필수 정보가 부족한 경우입니다.
이때 도구를 호출하지 않고 부족한 정보를 요청하는 응답을 해야 합니다.

[필수 조건]
도구 호출이 가능하려면 다음 중 하나를 만족해야 함:
- 성별 + 얼굴형이 모두 있음
- 퍼스널컬러가 있음

[예외 케이스 3가지]

1. 정보 전혀 없음 (성별X, 얼굴형X, 퍼스널컬러X)
   → 응답: "성별과 얼굴형 또는 퍼스널컬러를 알려주셔야 헤어스타일 추천이 가능합니다. 성별과 얼굴형 또는 퍼스널컬러를 알려주시겠어요?😊"

2. 성별만 있음 (성별O, 얼굴형X, 퍼스널컬러X)
   → 응답: "얼굴형을 알려주셔야 헤어스타일 추천이 가능합니다. 얼굴형을 알려주시겠어요?😊"

3. 얼굴형만 있음 (성별X, 얼굴형O, 퍼스널컬러X)
   → 응답: "성별을 알려주셔야 헤어스타일 추천이 가능합니다. 성별을 알려주시겠어요?😊"

[생성 규칙]
각 케이스별로 다양한 질의를 생성해주세요:
- 케이스1 (정보 없음): {num_samples // 3}개
- 케이스2 (성별만): {num_samples // 3}개  
- 케이스3 (얼굴형만): {num_samples // 3}개

다양한 말투와 표현을 사용해주세요:
- "추천해줘", "알려줘", "뭐가 좋아?", "어울릴까?"
- 반말, 존댓말, 이모지 등

[출력 형식]
JSON 배열로 출력. 각 항목:
{{
  "type": "no_info" | "gender_only" | "face_shape_only",
  "user": "사용자 질의",
  "assistant": "위에 정의된 응답 그대로 사용"
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
    "no_info": "성별과 얼굴형 또는 퍼스널컬러를 알려주셔야 헤어스타일 추천이 가능합니다. 성별과 얼굴형 또는 퍼스널컬러를 알려주시겠어요?😊",
    "gender_only": "얼굴형을 알려주셔야 헤어스타일 추천이 가능합니다. 얼굴형을 알려주시겠어요?😊",
    "face_shape_only": "성별을 알려주셔야 헤어스타일 추천이 가능합니다. 성별을 알려주시겠어요?😊"
}


def convert_to_training_format(samples: list) -> list:
    """생성된 샘플을 학습 데이터 형식으로 변환"""
    
    training_data = []
    
    for sample in samples:
        # 타입에 따라 고정 응답 사용 (GPT가 생성한 응답 대신)
        response = RESPONSE_MAP.get(sample["type"], sample["assistant"])
        
        training_sample = {
            "messages": [
                {"role": "system", "content": ""},
                {"role": "user", "content": sample["user"]},
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


def get_data(num_samples: int = 60, output_file: str = "no_image_exception.jsonl"):
    """메인 함수: 데이터 생성 → 변환 → 저장"""
    
    print(f"이미지 없는 추천 예외처리 샘플 {num_samples}개 생성 중...")
    
    # 1. GPT로 샘플 생성
    raw_samples = generate_no_image_exception_samples(num_samples)
    print(f"생성 완료: {len(raw_samples)}개")
    
    # 타입별 개수 확인
    type_counts = {}
    for sample in raw_samples:
        t = sample["type"]
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f"타입별 분포: {type_counts}")
    
    # 2. 학습 데이터 형식으로 변환
    training_data = convert_to_training_format(raw_samples)
    
    # 3. JSONL 저장
    save_to_jsonl(training_data, output_file)
    
    return training_data


if __name__ == "__main__":
    data = get_data(num_samples=60, output_file="samples/no_image_exception.jsonl")
    
    # 확인용 출력
    print("\n=== 샘플 미리보기 ===")
    for i, sample in enumerate(data[:6]):
        user_msg = sample['messages'][1]['content']
        assistant_msg = sample['messages'][2]['content']
        
        print(f"\n[{i+1}] User: {user_msg}")
        print(f"    Assistant: {assistant_msg[:50]}...")

## 실행 결과 예시
"""
이미지 없는 추천 예외처리 샘플 60개 생성 중...
생성 완료: 60개
타입별 분포: {'no_info': 20, 'gender_only': 20, 'face_shape_only': 20}
저장 완료: samples/no_image_exception.jsonl (60개 샘플)

=== 샘플 미리보기 ===

[1] User: 헤어스타일 추천해줘
    Assistant: 성별과 얼굴형 또는 퍼스널컬러를 알려주셔야 헤어스타일 추천이 가능합니다...

[2] User: 나한테 어울리는 머리 뭐야?
    Assistant: 성별과 얼굴형 또는 퍼스널컬러를 알려주셔야 헤어스타일 추천이 가능합니다...

[3] User: 여자인데 머리 추천해줘~
    Assistant: 얼굴형을 알려주셔야 헤어스타일 추천이 가능합니다...

[4] User: 남자인데 어떤 헤어스타일이 좋을까요?
    Assistant: 얼굴형을 알려주셔야 헤어스타일 추천이 가능합니다...

[5] User: 둥근 얼굴인데 어울리는 머리 추천해줘
    Assistant: 성별을 알려주셔야 헤어스타일 추천이 가능합니다...

[6] User: 사각턱인데 뭐가 좋아?
    Assistant: 성별을 알려주셔야 헤어스타일 추천이 가능합니다...
"""