import json
from openai import OpenAI

client = OpenAI()

def generate_no_image_recommendation_samples(num_samples: int = 50) -> list:
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

[생성 규칙]
다양한 조합으로 {num_samples}개 생성:
- 성별 + 얼굴형만 있는 경우
- 성별 + 얼굴형 + 계절이 있는 경우
- 성별 + 얼굴형 + 스타일 키워드가 있는 경우
- 성별 + 얼굴형 + 컬러 키워드가 있는 경우
- 퍼스널컬러만 있는 경우
- 퍼스널컬러 + 키워드가 있는 경우
- 모든 정보가 다 있는 경우

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
    "haircolor_keywords": "톤다운" (있는 경우만)
  }}
}}

arguments에는 질의에서 추출 가능한 파라미터만 포함하세요.
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


def convert_to_training_format(samples: list) -> list:
    """생성된 샘플을 학습 데이터 형식으로 변환"""
    
    training_data = []
    
    for i, sample in enumerate(samples):
        # arguments를 JSON 문자열로 변환
        arguments_str = json.dumps(sample["arguments"], ensure_ascii=False)
        
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
    with open(filename, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"저장 완료: {filename} ({len(data)}개 샘플)")


def get_data(num_samples: int = 50, output_file: str = "no_image_recommendation.jsonl"):
    """메인 함수: 데이터 생성 → 변환 → 저장"""
    
    print(f"이미지 없는 추천 샘플 {num_samples}개 생성 중...")
    
    # 1. GPT로 샘플 생성
    raw_samples = generate_no_image_recommendation_samples(num_samples)
    print(f"생성 완료: {len(raw_samples)}개")
    
    # 2. 학습 데이터 형식으로 변환
    training_data = convert_to_training_format(raw_samples)
    
    # 3. JSONL 저장
    save_to_jsonl(training_data, output_file)
    
    return training_data


if __name__ == "__main__":
    data = get_data(num_samples=50, output_file="samples/no_image_recommendation.jsonl")
    
    # 확인용 출력
    print("\n=== 샘플 미리보기 ===")
    for i, sample in enumerate(data[:3]):
        user_msg = sample['messages'][1]['content']
        tool_call = sample['messages'][2]['tool_calls'][0]
        args = tool_call['function']['arguments']
        
        print(f"\n[{i+1}] User: {user_msg}")
        print(f"    Tool: {tool_call['function']['name']}")
        print(f"    Args: {args}")


## 실행 결과 예시
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