import json
import re
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

# --------------------------------------------------
# JSON 안전 파서
# --------------------------------------------------
def safe_load_json(content: str):
    content = content.strip()

    # 코드블록 제거
    content = re.sub(r"^```json|^```", "", content)
    content = re.sub(r"```$", "", content)

    # stray comma 제거
    content = re.sub(r",\s*]", "]", content)
    content = re.sub(r",\s*}", "}", content)

    # 괄호 불일치 자동 보정
    if content.count("[") > content.count("]"):
        content += "]"
    if content.count("{") > content.count("}"):
        content += "}" * (content.count("{") - content.count("}"))

    return json.loads(content)


def generate_batch(batch_size: int):
    """100개 단위로 안정적으로 생성"""
    
    system_prompt = "You ONLY output valid JSON array. No explanation, no commentary."

    prompt = f"""
        헤어스타일 상담 챗봇 'HairAllYou'의 일반 질의 대응 데이터셋을 생성해주세요.

        총 {batch_size}개의 항목.
        각 항목은 JSON 배열 요소로 출력하세요.

        형식:
        {{
          "type": "...",
          "user": "...",
          "assistant": "..."
        }}

        생성 규칙은 다음과 같습니다:

        [카테고리 1: greeting]
        user 예: 안녕, 하이, 반가워
        assistant:
        "안녕하세요, 저는 헤어스타일과 관련된 상담을 도와주는 HairAllYou 챗봇🤖입니다. 어떤 것을 도와드릴까요?"

        [카테고리 2: irrelevant]
        user 예: 날씨, 주식, 요리, 코딩 등
        assistant:
        "저는 헤어스타일과 관련된 상담을 도와주는 HairAllYou 챗봇입니다. 헤어스타일에 대한 것만 답변해드릴 수 있어요🥲"

        [카테고리 3: function_info]
        assistant:
        "저는 헤어스타일에 관련된 상담을 도와드리는 챗봇입니다😊
        1. 헤어스타일 정보
        2. 어울리는 스타일 추천
        3. 이미지 분석
        4. 이미지 생성
        무엇부터 도와드릴까요?😊"

        [카테고리 4: hair_change]
        assistant:
        "헤어스타일을 변경하고 싶으시군요!😊 어울릴 스타일 추천을 위해 다음 정보를 알려주세요!
        - 성별
        - 얼굴형
        - 퍼스널컬러
        원하는 길이나 느낌도 함께 알려주세요!"

        [카테고리 5: hair_dissatisfaction]
        assistant:
        "헤어스타일이 마음에 들지 않으시다니 속상하시겠어요🥲 하지만 걱정하지 마세요! 다음번 실패하지 않도록 팁 알려드릴까요?"

        [카테고리 6: salon_tips]
        assistant:
        "미용실 팁을 알려드릴게요! 사진 준비, 모발 상태 공유, 요구사항 구체화, 중간 소통, 관리법 질문하기 등이 있어요😊"

        [카테고리 7: shop_restriction]
        assistant:
        "죄송하지만 특정 미용실이나 디자이너 추천은 어려워요🥲 헤어스타일 정보는 언제든지 도와드릴게요!😊"
    """

    for _ in range(3):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
        )
        content = response.choices[0].message.content.strip()

        try:
            return safe_load_json(content)
        except:
            continue

    raise ValueError("JSON 파싱 실패(3회).")


def generate_greeting_and_irrelevant_samples(num_samples: int = 100):
    results = []
    batch_size = 20

    while len(results) < num_samples:
        needed = min(batch_size, num_samples - len(results))
        batch = generate_batch(needed)
        results.extend(batch)

    return results[:num_samples]


def convert_to_training_format(samples: list) -> list:
    training_data = []
    for sample in samples:
        training_data.append({
            "messages": [
                {"role": "system", "content": ""},
                {"role": "user", "content": sample["user"]},
                {"role": "assistant", "content": sample["assistant"]},
            ]
        })
    return training_data


def save_to_jsonl(data: list, filename: str):
    with open(filename, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"저장 완료: {filename} ({len(data)}개)")


def get_data(num_samples: int = 100, output_file: str = "greeting_irrelevant.jsonl"):
    print(f"{num_samples}개 생성 시작...(gpt-4o-mini)")

    raw = generate_greeting_and_irrelevant_samples(num_samples)
    print(f"생성 완료: {len(raw)}개")

    training = convert_to_training_format(raw)
    save_to_jsonl(training, output_file)

    return training


if __name__ == "__main__":
    get_data(num_samples=100, output_file="finetuning/samples/greeting_irrelevant.jsonl")
