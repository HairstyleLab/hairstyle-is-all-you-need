import json
from openai import OpenAI

client = OpenAI()

def qa_01(num_samples: int = 10) -> list:
    """인사 및 관련없는 질문 샘플 생성"""
    
    prompt = f"""
        헤어스타일 추천 챗봇 'HairAllYou'의 학습 데이터를 생성해주세요.

        [카테고리 1: 인사]
        사용자가 인사를 하면 다음과 같이 응답:
        "안녕하세요, 저는 헤어스타일과 관련된 상담을 도와주는 HairAllYou 챗봇🤖입니다. 어떤 것을 도와드릴까요?"

        [카테고리 2: 관련없는 질문]  
        사용자가 헤어스타일과 관련 없는 질문을 하면 다음과 같이 응답:
        "저는 헤어스타일과 관련된 상담을 도와주는 HairAllYou 챗봇입니다. 헤어스타일에 대한 것만 질문해주세요😊"

        다양한 사용자 질의를 {num_samples}개 생성해주세요.
        - 인사: {num_samples // 2}개 (안녕, 하이, 반가워, 안녕하세요 등 다양한 표현)
        - 관련없는 질문: {num_samples // 2}개 (날씨, 주식, 음식, 여행, 게임, 연애 등 다양한 주제)

        [출력 형식]
        JSON 배열로 출력. 각 항목은 다음 형식:
        {{"type": "greeting" 또는 "irrelevant", "user": "사용자 질의", "assistant": "챗봇 응답"}}

        JSON 배열만 출력하고 다른 설명은 하지 마세요.
        """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,
    )
    
    # JSON 파싱
    content = response.choices[0].message.content
    # ```json 등 마크다운 제거
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1]
    if content.endswith("```"):
        content = content.rsplit("```", 1)[0]
    
    samples = json.loads(content)
    return samples

def convert_to_training_format(samples: list) -> list:
    """생성된 샘플을 학습 데이터 형식으로 변환"""
    
    training_data = []
    
    for sample in samples:
        training_sample = {
            "messages": [
                {"role": "system", "content": ""},
                {"role": "user", "content": sample["user"]},
                {"role": "assistant", "content": sample["assistant"]}
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


def get_data(num_samples: int = 50, output_file: str = "greeting_irrelevant.jsonl"):
    """메인 함수: 데이터 생성 → 변환 → 저장"""
    
    print(f"샘플 {num_samples}개 생성 중...")
    
    # 1. GPT로 샘플 생성
    raw_samples = generate_greeting_and_irrelevant_samples(num_samples)
    print(f"생성 완료: {len(raw_samples)}개")
    
    # 2. 학습 데이터 형식으로 변환
    training_data = convert_to_training_format(raw_samples)
    
    # 3. JSONL 저장
    save_to_jsonl(training_data, output_file)
    
    return training_data


if __name__ == "__main__":
    data = get_data(num_samples=50, output_file="samples/greeting_irrelevant.jsonl")
    
    # 확인용 출력
    print("\n=== 샘플 미리보기 ===")
    for i, sample in enumerate(data[:3]):
        print(f"\n[{i+1}] {sample['messages'][1]['content']}")
        print(f"    → {sample['messages'][2]['content']}")
