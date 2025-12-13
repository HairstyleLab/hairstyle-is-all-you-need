import json
from openai import OpenAI

client = OpenAI()

def generate_web_search_samples(num_samples: int = 50) -> list:
    """웹 검색 도구 호출 샘플 생성"""
    
    prompt = f"""
헤어스타일 추천 챗봇의 학습 데이터를 생성해주세요.

[시나리오]
사용자가 헤어스타일 트렌드, 유행, 최신 정보를 물어보는 경우입니다.
이때 web_search_tool을 호출해야 합니다.

[web_search_tool 호출 조건]
- "요즘", "최근", "유행", "트렌드", "인기" 등의 키워드가 포함된 질문
- 특정 연예인/셀럽 헤어스타일 질문
- 계절별 트렌드 질문
- 특정 스타일의 유행 여부 질문

[검색 쿼리 생성 규칙]
- 항상 "2025"를 포함
- 항상 "한국"을 포함
- 핵심 키워드 포함 (남자/여자, 헤어스타일/염색/펌 등)
- 자연스러운 검색어로 구성

[생성할 질문 유형]
1. 일반 트렌드: "요즘 유행하는 머리 뭐야?"
2. 성별 특정: "남자 헤어스타일 트렌드 알려줘"
3. 스타일 특정: "숏컷 유행하는 스타일 있어?"
4. 염색 트렌드: "요즘 인기있는 염색 색상 뭐야?"
5. 계절 트렌드: "여름에 유행하는 머리 알려줘"
6. 펌 트렌드: "요즘 남자 펌 뭐가 인기야?"
7. 연예인 스타일: "아이유 헤어스타일 뭐야?" (이런 류)
8. 비교 질문: "레이어드컷이랑 허쉬컷 중에 뭐가 더 유행해?"

다양한 말투로 {num_samples}개 생성해주세요.

[출력 형식]
JSON 배열로 출력. 각 항목:
{{
  "user": "사용자 질의",
  "query": "2025 ... 한국" (검색 쿼리)
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


def convert_to_training_format(samples: list) -> list:
    """생성된 샘플을 학습 데이터 형식으로 변환"""
    
    training_data = []
    
    for i, sample in enumerate(samples):
        arguments_str = json.dumps({"query": sample["query"]}, ensure_ascii=False)
        
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
                                "name": "web_search_tool",
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


def get_data(num_samples: int = 50, output_file: str = "web_search.jsonl"):
    """메인 함수: 데이터 생성 → 변환 → 저장"""
    
    print(f"웹 검색 샘플 {num_samples}개 생성 중...")
    
    # 1. GPT로 샘플 생성
    raw_samples = generate_web_search_samples(num_samples)
    print(f"생성 완료: {len(raw_samples)}개")
    
    # 2. 학습 데이터 형식으로 변환
    training_data = convert_to_training_format(raw_samples)
    
    # 3. JSONL 저장
    save_to_jsonl(training_data, output_file)
    
    return training_data


if __name__ == "__main__":
    data = get_data(num_samples=50, output_file="samples/web_search.jsonl")
    
    # 확인용 출력
    print("\n=== 샘플 미리보기 ===")
    for i, sample in enumerate(data[:5]):
        user_msg = sample['messages'][1]['content']
        tool_call = sample['messages'][2]['tool_calls'][0]
        args = json.loads(tool_call['function']['arguments'])
        
        print(f"\n[{i+1}] User: {user_msg}")
        print(f"    Query: {args['query']}")

"""
## 실행 결과 예시

웹 검색 샘플 50개 생성 중...
생성 완료: 50개
저장 완료: samples/web_search.jsonl (50개 샘플)

=== 샘플 미리보기 ===

[1] User: 요즘 유행하는 남자 헤어스타일 뭐야?
    Query: 2025 남자 헤어스타일 트렌드 한국

[2] User: 여자 숏컷 트렌드 알려줘~
    Query: 2025 여자 숏컷 트렌드 한국

[3] User: 최근 인기있는 염색 색상 추천해줘
    Query: 2025 염색 인기 색상 트렌드 한국

[4] User: 겨울에 어울리는 펌 스타일 뭐가 유행이야?
    Query: 2025 겨울 펌 스타일 트렌드 한국

[5] User: 요즘 남자들 무슨 펌 많이 해?
    Query: 2025 남자 펌 인기 트렌드 한국
"""