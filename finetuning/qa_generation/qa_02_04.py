import json
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

def generate_web_search_samples(num_samples: int = 10) -> list:
    """웹 검색 도구 호출 샘플 생성"""
    
    prompt = f"""
        헤어스타일 추천/정보/트렌드/관리/제품 추천 챗봇의 학습 데이터를 생성해주세요.

        [시나리오]
        사용자가 헤어스타일과 관련된 '정확한 정보', '팩트 기반 설명', 
        '차이점·특징', '스타일 정의', '유명인 스타일', '제품·관리법 정보', 
        '최신 트렌드' 등을 요구하는 경우입니다.
        이러한 질문은 대부분 외부 정보 확인이 필요하므로 web_search_tool을 호출합니다.

        [web_search_tool 호출 조건]
        아래 중 하나라도 해당되면 반드시 검색 도구를 호출해야 합니다.
        - 최신/최근/요즘/유행/트렌드/인기 관련 질문
        - 특정 헤어스타일의 정의, 특징, 차이점 요청  
        - 특정 펌/염색/커트 명칭에 대한 정보 질문  
        - 특정 유명인/셀럽 헤어스타일 질문
        - 계절별 스타일/관리/추천 질문
        - 헤어 관리 팁/관리 루틴/손상 케어 등 정보 요구
        - 헤어 제품 추천, 제품 비교, 성능 정보 질문
        - 가격대, 시술 명칭, 난이도, 유지기간 등 정보 확인

        즉, **사실성·정보 기반 답변이 필요한 질문은 모두 web_search_tool 호출 대상입니다.**

        [검색 쿼리 생성 규칙 — 일반화된 형태]
        - 사용자의 질문 의도를 가장 잘 반영한 핵심 검색어 중심으로 구성
        - 필요하다면 성별/스타일명/시술명/연예인 이름 등 맥락 키워드를 포함
        - “한국”, “2025”, “최신”, “정보”, “비교”, “차이”, “관리법” 등은  
        **질문을 더 정확하게 검색할 수 있을 경우에만 선택적으로 포함**
        - 불필요하게 길지 않고 실제 검색 엔진에 입력할 법한 자연스러운 문구로 구성
        - 완전한 문장일 필요 없음: 핵심 키워드 위주로 구성해도 됨

        예:  
        - “레이어드컷 허쉬컷 차이”  
        - “빌드펌 특징 한국”  
        - “여름 머리 관리법 최신”  
        - “카리나 헤어스타일 정보”  
        - “헤어 오일 추천 손상모”

        [생성할 질문 유형 — 확장 버전]
        1. 트렌드 질문  
        2. 스타일 정의 질문  
        3. 차이점 질문  
        4. 성별 기반 스타일 정보  
        5. 염색 정보·비교  
        6. 펌 정보  
        7. 계절별 스타일/관리  
        8. 유명인 스타일 정보  
        9. 제품 추천  
        10. 제품 비교  
        11. 관리 팁  
        12. 가격/시술 정보  
        13. 정보 확인형 질문  

        위 유형을 섞어서 다양한 말투로 {num_samples}개 생성해주세요.

        [출력 형식]
        JSON 배열로 출력. 각 항목:
        {{
        "user": "사용자 질의",
        "query": "(검색 쿼리)"
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
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(filename, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"저장 완료: {filename} ({len(data)}개 샘플)")


def get_data(num_samples: int = 50, output_file: str = "finetuning/samples/qa_02_04.jsonl"):
    """메인 함수: 데이터 생성 → 변환 → 저장"""
    
    print(f"웹 검색 샘플 {num_samples}개 생성 중...")
    
    raw_samples = generate_web_search_samples(num_samples)
    print(f"생성 완료: {len(raw_samples)}개")
    
    training_data = convert_to_training_format(raw_samples)
    
    save_to_jsonl(training_data, output_file)
    
    return training_data


if __name__ == "__main__":
    data = get_data(num_samples=100, output_file="finetuning/samples/qa_02_04.jsonl")
    
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