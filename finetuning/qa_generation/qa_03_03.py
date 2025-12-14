import json
import random
import base64
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()


MALE_CUTS = [
    "가일컷", "댄디컷", "드랍컷", "리젠트컷", "리프컷", "버즈컷", "슬릭백컷",
    "아이브리그컷", "울프컷", "크롭컷", "크루컷", "투블럭컷", "페이드컷",
    "포마드컷", "필러스컷", "하이앤타이트컷", "가르마펌"
]

MALE_PERMS = [
    "가일펌", "내추럴펌", "댄디펌", "리젠트펌", "리프펌", "베이비펌", "볼륨펌",
    "쉐도우펌", "스왈로펌", "애즈펌", "웨이브펌", "크리드펌", "포마드펌", "히피펌"
]

FEMALE_CUTS = [
    "레이어드컷", "리프컷", "머쉬룸컷", "뱅헤어", "보브컷", "샤기컷",
    "원랭스컷", "픽시컷", "허쉬컷", "히메컷"
]

FEMALE_PERMS = [
    "CS컬펌", "C컬펌", "S컬펌", "글램펌", "내츄럴펌", "디지털펌", "러블리펌",
    "레이어드펌", "루즈펌", "리프펌", "물결펌", "믹스펌", "바디펌", "발롱펌",
    "볼드펌", "볼륨매직", "볼륨펌", "빌드펌", "셋팅펌", "스파이럴펌", "에어펌",
    "젤리펌", "지젤펌", "쿠션펌", "텍스처펌", "퍼피베이비펌", "허쉬펌", "히피펌"
]

HAIR_COLORS = [
    "골드브라운", "다크브라운", "레드브라운", "레드와인", "로즈골드", "마르살라",
    "마호가니", "밀크브라운", "베이지브라운", "블루블랙", "애쉬그레이", "애쉬바이올렛",
    "애쉬베이지", "애쉬브라운", "애쉬블론드", "애쉬블루", "애쉬카키", "애쉬퍼플",
    "오렌지브라운", "올리브브라운", "초코브라운", "카키브라운", "쿠퍼브라운", "핑크브라운"
]

MALE_LENGTHS = ["숏", "미디엄", "장발"]
FEMALE_LENGTHS = ["숏", "단발", "중단발", "미디엄", "장발"]

LENGTH_EXPRESSIONS = {
    "male": {
        "숏": ["짧게", "숏으로", "짧은 머리로", "귀 위로", "목 드러나게", "시원하게 짧게", "숏컷으로"],
        "미디엄": ["중간 길이로", "미디엄으로", "귀 아래 정도로", "턱선 길이로", "가르마 스타일로"],
        "장발": ["길게", "장발로", "어깨까지", "긴머리로", "웨이브 넣을 수 있게 길게"]
    },
    "female": {
        "숏": ["짧게", "숏으로", "귀 아래로", "숏컷으로", "짧은 머리로"],
        "단발": ["단발로", "턱선 길이로", "턱 아래로", "단발머리로"],
        "중단발": ["중단발로", "어깨 닿는 길이로", "어깨선으로", "어깨 위로"],
        "미디엄": ["미디엄으로", "쇄골 길이로", "세미롱으로", "어깨 아래로"],
        "장발": ["길게", "장발로", "가슴까지", "긴머리로", "롱헤어로", "쇄골 아래로"]
    }
}


def generate_exception_queries(num_samples=10):
    """
    2가지 이미지 예외 케이스 생성:
      - no_face: 얼굴 없는 이미지 (풍경, 사물, 음식 등)
      - multi_face: 2명 이상 나온 이미지

    중요: 사용자는 정상적인 헤어스타일 변환 요청을 하지만,
          이미지에 문제가 있어서 예외 응답이 나와야 하는 상황
    """

    prompt = f"""
헤어스타일 이미지 생성 챗봇의 예외처리 학습 데이터를 만듭니다.

[시나리오]
사용자가 자신의 사진을 업로드하고 정상적으로 헤어스타일 변환을 요청합니다.
하지만 업로드된 이미지에 문제가 있어서 모델이 예외 응답을 해야 하는 상황입니다.

[예외 케이스 2가지]
1. no_face: 이미지에 얼굴이 없음 (풍경, 음식, 사물, 동물 사진 등)
   → 응답: "얼굴이 포함된 이미지를 첨부하셔야 이미지를 만들 수 있습니다🥲 확인 후 다른 사진을 업로드해주세요."

2. multi_face: 이미지에 2명 이상의 얼굴이 있음 (단체사진, 커플사진 등)
   → 응답: "이 이미지에는 2명 이상의 얼굴이 포함되어 있습니다🥲 한 명만 나온 이미지를 업로드해주세요."

[중요: 사용자 질의 특징]
- 사용자는 이미지에 문제가 있다는 걸 모릅니다
- 정상적인 헤어스타일 변환 요청을 합니다
- "히메컷으로 바꿔줘", "레드와인 컬러로 염색해줘", "단발로 잘라줘" 같은 일반적인 요청
- 절대로 "얼굴이 없는데", "사진에 두명이 나왔는데" 같은 표현 사용 금지

[질의 예시]
- "이 사진으로 히메컷에 레드와인 컬러 적용해줘"
- "허쉬펌으로 바꿔주세요"
- "애쉬브라운으로 염색하고 싶어요"
- "단발로 잘라주고 볼륨펌 넣어줘"
- "이 얼굴에 울프컷 어울릴까? 적용해봐줘"

[생성 규칙]
- 총 {num_samples}개 생성 (no_face, multi_face 균등 분배)
- 반말/존댓말 섞기, 이모지 가끔 사용
- 표현 다양화: "이 사진으로", "내 얼굴에", "이 이미지로", "바꿔줘", "적용해줘", "변환해줘" 등

[출력: JSON 배열만]
{{
  "type": "no_face" | "multi_face",
  "user": "사용자의 정상적인 헤어스타일 변환 요청"
}}

JSON 배열만 출력하세요.
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,
    )

    content = response.choices[0].message.content.strip()

    if content.startswith("```"):
        lines = content.split("\n")
        content = "\n".join(lines[1:-1]) if len(lines) > 2 else content
        content = content.strip()

    if content.startswith("json"):
        content = content[4:].strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"[경고] JSON 파싱 에러 (generate_exception_queries): {e}")
        print(f"[디버그] 첫 100자: {content[:100]}")

        bracket_count = 0
        end_idx = -1
        for i, char in enumerate(content):
            if char == '[':
                bracket_count += 1
            elif char == ']':
                bracket_count -= 1
                if bracket_count == 0:
                    end_idx = i + 1
                    break

        if end_idx > 0:
            valid_content = content[:end_idx]
            print(f"[디버그] 유효한 부분만 추출: {len(valid_content)}자")
            return json.loads(valid_content)
        else:
            raise


def generate_normal_queries_with_length(num_samples=10):
    """
    2가지 스타일/컬러 예외 케이스 생성:
      - unsupported_style: 지원하지 않는 헤어스타일/컬러 이름 요청
      - missing_style: 스타일/컬러를 아예 지정하지 않음

    중요: 사용자가 이미지 변환을 요청하지만,
          스타일명이 지원 목록에 없거나 아무것도 안 적어서 예외 응답 필요
    """

    male_styles = ", ".join(MALE_CUTS + MALE_PERMS)
    female_styles = ", ".join(FEMALE_CUTS + FEMALE_PERMS)
    colors = ", ".join(HAIR_COLORS)

    prompt = f"""
헤어스타일 이미지 생성 챗봇의 스타일/컬러 예외처리 학습 데이터를 만듭니다.

[시나리오]
사용자가 자신의 얼굴 사진(정상)을 업로드하고 헤어스타일 변환을 요청합니다.
하지만 스타일명이 지원 목록에 없거나 아무것도 지정하지 않아서 예외 응답이 필요한 상황입니다.

[예외 케이스 2가지]
1. unsupported_style: 지원하지 않는 헤어스타일/컬러 이름 요청
   - 사용자가 존재하지 않거나 지원 안되는 스타일명 요청
   - 또는 "청순하게", "시크하게", "멋있게" 같이 구체적인 스타일명이 아닌 형용사/추상적 표현
   → 응답: "요청하신 스타일은 지원하지 않습니다🥲\\n\\n지원 가능한 스타일/컬러 목록을 확인하고 다시 요청해주세요!"

2. missing_style: 스타일/컬러를 아예 지정하지 않음
   - "이 사진으로 머리 바꿔줘", "헤어스타일 바꿔주세요", "변신시켜줘" 같이 구체적인 스타일/컬러 없음
   → 응답: "구체적인 헤어스타일이나 컬러를 지정해주세요!"

[지원 가능한 목록]
- 남자 스타일: {male_styles}
- 여자 스타일: {female_styles}
- 컬러: {colors}

[질의 예시]
unsupported_style (지원 안되는 이름):
- "태슬펌으로 바꿔줘" (존재하지 않음)
- "그린컬러로 염색해줘" (지원 안함)
- "청순한 느낌으로 바꿔줘" (추상적 표현)
- "고대풍 헤어스타일로" (지원 안함)
- "네온 핑크로 염색해주세요" (지원 안함)
- "시크하고 멋있게 해줘" (형용사만)
- "엘사 머리로" (캐릭터 이름)
- "버섯 컷으로" (오타/다른 이름)

missing_style (스타일/컬러 없음):
- "이 사진으로 머리 바꿔줘"
- "헤어스타일 변신시켜주세요"
- "이미지 바꿔줄래?"
- "머리 좀 바꾸고 싶어"
- "다른 스타일로 해줘"
- "얼굴 좀 바꿔봐줘"
- "변신 부탁해"

[생성 규칙]
- 총 {num_samples}개 생성 (unsupported_style, missing_style 균등 분배)
- 반말/존댓말 섞기, 이모지 가끔 사용
- unsupported_style: 실제로 없거나 지원 안되는 이름, 형용사, 추상적 표현 사용 (창의적으로!)
- missing_style: 구체적 스타일명 절대 포함 금지, "머리", "스타일", "헤어" 같은 일반 명사만

[출력: JSON 배열만]
{{
  "type": "unsupported_style" | "missing_style",
  "user": "사용자 질의"
}}

JSON 배열만 출력하세요.
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,
    )

    content = response.choices[0].message.content.strip()

    if content.startswith("```"):
        lines = content.split("\n")
        content = "\n".join(lines[1:-1]) if len(lines) > 2 else content
        content = content.strip()

    if content.startswith("json"):
        content = content[4:].strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"[경고] JSON 파싱 에러 (generate_normal_queries_with_length): {e}")
        print(f"[디버그] 첫 100자: {content[:100]}")

        bracket_count = 0
        end_idx = -1
        for i, char in enumerate(content):
            if char == '[':
                bracket_count += 1
            elif char == ']':
                bracket_count -= 1
                if bracket_count == 0:
                    end_idx = i + 1
                    break

        if end_idx > 0:
            valid_content = content[:end_idx]
            print(f"[디버그] 유효한 부분만 추출: {len(valid_content)}자")
            return json.loads(valid_content)
        else:
            raise


RESPONSE_MAP = {
    "no_face":
        "얼굴이 포함된 이미지를 첨부하셔야 이미지를 만들 수 있습니다🥲 확인 후 다른 사진을 업로드해주세요.",

    "multi_face":
        "이 이미지에는 2명 이상의 얼굴이 포함되어 있습니다🥲 한 명만 나온 이미지를 업로드해주세요.",

    "unsupported_style":
        "죄송합니다🥲 요청하신 헤어스타일/컬러는 현재 지원되지 않습니다. 아래 옵션 목록에서 선택해 다시 시도해주세요.\n\n"
        "**지원 스타일**\n"
        "남자 컷: 가일컷, 댄디컷, 드랍컷, 리젠트컷, 리프컷, 버즈컷, 슬릭백컷, 아이브리그컷, 울프컷, 크롭컷, 크루컷, 투블럭컷, 페이드컷, 포마드컷, 필러스컷, 하이앤타이트컷, 가르마펌\n"
        "남자 펌: 가일펌, 내추럴펌, 댄디펌, 리젠트펌, 리프펌, 베이비펌, 볼륨펌, 쉐도우펌, 스왈로펌, 애즈펌, 웨이브펌, 크리드펌, 포마드펌, 히피펌\n"
        "여자 컷: 레이어드컷, 리프컷, 머쉬룸컷, 뱅헤어, 보브컷, 샤기컷, 원랭스컷, 픽시컷, 허쉬컷, 히메컷\n"
        "여자 펌: CS컬펌, C컬펌, S컬펌, 글램펌, 내츄럴펌, 디지털펌, 러블리펌, 레이어드펌, 루즈펌, 리프펌, 물결펌, 믹스펌, 바디펌, 발롱펌, 볼드펌, 볼륨매직, 볼륨펌, 빌드펌, 셋팅펌, 스파이럴펌, 에어펌, 젤리펌, 지젤펌, 쿠션펌, 텍스처펌, 퍼피베이비펌, 허쉬펌, 히피펌\n\n"
        "**지원 컬러**\n"
        "골드브라운, 다크브라운, 레드브라운, 레드와인, 로즈골드, 마르살라, 마호가니, 밀크브라운, 베이지브라운, 블루블랙, 애쉬그레이, 애쉬바이올렛, 애쉬베이지, 애쉬브라운, 애쉬블론드, 애쉬블루, 애쉬카키, 애쉬퍼플, 오렌지브라운, 올리브브라운, 초코브라운, 카키브라운, 쿠퍼브라운, 핑크브라운",

    "missing_style":
        "어떤 헤어스타일이나 헤어컬러로 변경하고 싶으신가요? 원하시는 스타일이나 컬러를 말씀해주세요😊"
}


def build_tool_call(sample):
    """정상 케이스에 대한 tool_call 형식 생성"""
    params = {}

    if sample.get("hairstyle"):
        params["hairstyle"] = sample["hairstyle"]
    if sample.get("haircolor"):
        params["haircolor"] = sample["haircolor"]
    if sample.get("hairlength"):
        params["hairlength"] = sample["hairlength"]

    return {
        "name": "hairstyle_generation_tool",
        "parameters": params
    }


def convert_exception_to_training_format(samples):
    """예외 케이스 → 학습 포맷 (이미지는 build_dataset.py에서 매칭)"""
    training_data = []

    for s in samples:
        stype = s["type"]
        assistant_reply = RESPONSE_MAP[stype]

        if stype == "no_face":
            image_type = "no_face"
        elif stype == "multi_face":
            image_type = "multi_face"
        else:
            image_type = "normal"

        training_data.append({
            "messages": [
                {"role": "system", "content": ""},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": s["user"]},
                        {"type": "image_url", "image_url": {"url": ""}}
                    ]
                },
                {"role": "assistant", "content": assistant_reply}
            ],
            "image_type": image_type
        })

    return training_data


def convert_normal_to_training_format(samples):
    """
    정상(예외) 케이스 → 학습 포맷
    unsupported_style, missing_style은 텍스트 응답만
    이미지는 build_dataset.py에서 매칭
    """
    training_data = []

    for i, s in enumerate(samples):
        stype = s["type"]
        assistant_reply = RESPONSE_MAP[stype]

        training_data.append({
            "messages": [
                {"role": "system", "content": ""},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": s["user"]},
                        {"type": "image_url", "image_url": {"url": ""}}
                    ]
                },
                {"role": "assistant", "content": assistant_reply}
            ],
            "image_type": "normal"
        })

    return training_data


def save_jsonl(data, filename):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[저장 완료] {filename} ({len(data)}개)")


def get_data(
    num_exception_samples: int = 10,
    num_normal_samples: int = 10,
    output_exception: str = "finetuning/samples/qa_03_03_exception.jsonl",
    output_normal: str = "finetuning/samples/qa_03_03_normal.jsonl",
    output_combined: str = "finetuning/samples/qa_03_03_combined.jsonl"
):
    """메인 함수: 텍스트 질의-응답 생성 (이미지는 build_dataset.py에서 매칭)"""

    print("\n" + "="*60)
    print("qa_03_03: 이미지 생성 예외처리 데이터셋 생성")
    print("="*60)

    print(f"\n### 1. 이미지 예외 케이스 질의 생성 (GPT) - {num_exception_samples}개")
    exception_samples = generate_exception_queries(num_exception_samples)
    print(f"[생성 완료] 이미지 예외 케이스 {len(exception_samples)}개")

    print(f"\n### 2. 스타일/컬러 예외 케이스 질의 생성 (GPT) - {num_normal_samples}개")
    normal_samples = generate_normal_queries_with_length(num_normal_samples)
    print(f"[생성 완료] 스타일/컬러 예외 케이스 {len(normal_samples)}개")

    print("\n### 3. 학습 포맷 변환")
    exception_data = convert_exception_to_training_format(exception_samples)
    normal_data = convert_normal_to_training_format(normal_samples)

    print("\n### 4. JSONL 저장")
    save_jsonl(exception_data, output_exception)
    save_jsonl(normal_data, output_normal)

    combined_data = exception_data + normal_data
    random.shuffle(combined_data)
    save_jsonl(combined_data, output_combined)

    print("\n### 5. 생성 통계")
    print(f"  - 이미지 예외 케이스 (no_face, multi_face): {len(exception_data)}개")
    print(f"  - 스타일/컬러 예외 케이스 (unsupported_style, missing_style): {len(normal_data)}개")
    print(f"  - 전체 통합: {len(combined_data)}개")

    exception_type_counts = {}
    for s in exception_samples:
        t = s.get("type", "unknown")
        exception_type_counts[t] = exception_type_counts.get(t, 0) + 1

    print("\n  [이미지 예외 케이스 유형별]")
    for t, c in sorted(exception_type_counts.items()):
        print(f"    - {t}: {c}개")

    type_counts = {}
    for s in normal_samples:
        t = s.get("type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    print("\n  [스타일/컬러 예외 케이스 유형별]")
    for t, c in sorted(type_counts.items()):
        print(f"    - {t}: {c}개")

    return {
        "exception": exception_data,
        "normal": normal_data,
        "combined": combined_data
    }


if __name__ == "__main__":
    data = get_data(
        num_exception_samples=100,
        num_normal_samples=100,
        output_exception="finetuning/samples/qa_03_03_exception.jsonl",
        output_normal="finetuning/samples/qa_03_03_normal.jsonl",
        output_combined="finetuning/samples/qa_03_03.jsonl"
    )