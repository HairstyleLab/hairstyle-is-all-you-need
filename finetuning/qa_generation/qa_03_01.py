import json
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

def generate_no_image_exception_samples(num_samples: int = 10) -> list:
    """이미지 없는 추천 예외처리 샘플 생성 (복잡한 질의 포함)"""

    batch_size = 50
    all_samples = []

    remaining = num_samples
    while remaining > 0:
        current_batch = min(batch_size, remaining)

        prompt = f"""
헤어스타일 추천 챗봇의 학습 데이터 {current_batch}개를 생성해주세요.

[시나리오]
사용자가 이미지 없이 헤어스타일 추천을 요청하지만 필수 정보가 부족한 경우입니다.

[예외 케이스 3가지]
1. no_info: 성별, 얼굴형, 퍼스널컬러 모두 없음
2. gender_only: 성별만 있음 (얼굴형, 퍼스널컬러 없음)
3. face_shape_only: 얼굴형만 있음 (성별, 퍼스널컬러 없음)

[생성 규칙]
- 총 {num_samples}개 생성 (각 케이스 균등 분배)
- 각 케이스별로 복잡도 다양하게:
  * simple (30%): "헤어스타일 추천해줘", "어떤 머리가 좋아?"
  * medium (40%): 상황 1-2개 포함 (예: "여름이라 시원한 머리 추천해줘")
  * complex (30%): 여러 조건과 상세 맥락 (예: "다음주 면접인데 깔끔하면서도 너무 딱딱하지 않고 세련된 느낌으로 추천해줘")

[질의 특징]
- gender_only: 반드시 성별 포함 ("남자인데", "여자인데")
- face_shape_only: 반드시 얼굴형 포함 ("둥근 얼굴", "계란형", "긴 얼굴", "사각턱")
- no_info: 성별, 얼굴형, 퍼스널컬러 절대 포함 금지
- 다양한 말투: 반말/존댓말, 이모지 가끔 사용
- 맥락 추가: 계절, 이벤트(면접/소개팅/결혼식), 직업, 고민사항 등

[출력 형식]
{{
  "type": "no_info" | "gender_only" | "face_shape_only",
  "user": "사용자 질의",
  "complexity": "simple" | "medium" | "complex"
}}

JSON 배열만 출력하세요.
"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.95,
        )

        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
        if content.endswith("```"):
            content = content.rsplit("```", 1)[0]

        batch_samples = json.loads(content)
        all_samples.extend(batch_samples)
        remaining -= len(batch_samples)

        print(f"  배치 생성: {len(batch_samples)}개 (누적: {len(all_samples)}개)")

    return all_samples


RESPONSE_MAP = {
    "no_info": "성별과 얼굴형 또는 퍼스널컬러를 알려주셔야 헤어스타일 추천이 가능합니다. 성별과 얼굴형 또는 퍼스널컬러를 알려주시겠어요?😊",
    "gender_only": "얼굴형을 알려주셔야 헤어스타일 추천이 가능합니다. 얼굴형을 알려주시겠어요?😊",
    "face_shape_only": "성별을 알려주셔야 헤어스타일 추천이 가능합니다. 성별을 알려주시겠어요?😊"
}


def convert_to_training_format(samples: list) -> list:
    """생성된 샘플을 학습 데이터 형식으로 변환"""
    
    training_data = []
    
    for sample in samples:
        response = RESPONSE_MAP.get(sample["type"])
        
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
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"저장 완료: {filename} ({len(data)}개 샘플)")


def analyze_samples(samples: list):
    """생성된 샘플 분석"""
    
    type_counts = {"no_info": 0, "gender_only": 0, "face_shape_only": 0}
    complexity_counts = {"simple": 0, "medium": 0, "complex": 0}
    
    type_complexity = {
        "no_info": {"simple": 0, "medium": 0, "complex": 0},
        "gender_only": {"simple": 0, "medium": 0, "complex": 0},
        "face_shape_only": {"simple": 0, "medium": 0, "complex": 0}
    }
    
    for sample in samples:
        t = sample["type"]
        c = sample.get("complexity", "medium")
        
        type_counts[t] += 1
        complexity_counts[c] += 1
        type_complexity[t][c] += 1
    
    total = len(samples)
    print("\n=== 샘플 분석 ===")
    print(f"총 샘플 수: {total}개")
    
    print("\n[예외 타입 분포]")
    for key, count in type_counts.items():
        pct = (count / total * 100) if total > 0 else 0
        print(f"  {key}: {count}개 ({pct:.1f}%)")
    
    print("\n[복잡도 분포]")
    for key, count in complexity_counts.items():
        pct = (count / total * 100) if total > 0 else 0
        print(f"  {key}: {count}개 ({pct:.1f}%)")
    
    print("\n[타입별 복잡도 분포]")
    for t, complexities in type_complexity.items():
        print(f"  {t}:")
        for c, count in complexities.items():
            print(f"    - {c}: {count}개")
    
    lengths = [len(s["user"]) for s in samples]
    print(f"\n[질의 길이]")
    print(f"  최소: {min(lengths)}자")
    print(f"  최대: {max(lengths)}자")
    print(f"  평균: {sum(lengths)/len(lengths):.1f}자")


def get_data(num_samples: int = 10, output_file: str = "finetuning/samples/qa_03_01.jsonl"):
    """메인 함수: 데이터 생성 → 변환 → 저장"""
    
    print(f"이미지 없는 추천 예외처리 샘플 {num_samples}개 생성 중...")
    
    raw_samples = generate_no_image_exception_samples(num_samples)
    print(f"생성 완료: {len(raw_samples)}개")
    
    analyze_samples(raw_samples)
    
    training_data = convert_to_training_format(raw_samples)
    
    save_to_jsonl(training_data, output_file)
    
    return training_data, raw_samples


if __name__ == "__main__":
    training_data, raw_samples = get_data(
        num_samples=100,
        output_file="finetuning/samples/qa_03_01.jsonl"
    )
    
    print("\n=== 타입별 & 복잡도별 샘플 미리보기 ===")
    
    for exception_type in ["no_info", "gender_only", "face_shape_only"]:
        print(f"\n{'='*60}")
        print(f"[{exception_type.upper()}]")
        print(f"응답: {RESPONSE_MAP[exception_type][:50]}...")
        
        type_samples = [s for s in raw_samples if s["type"] == exception_type]
        
        for complexity in ["simple", "medium", "complex"]:
            filtered = [s for s in type_samples if s.get("complexity") == complexity]
            if filtered:
                print(f"\n  [{complexity}]")
                for sample in filtered[:2]:
                    text = sample['user']
                    if len(text) > 70:
                        text = text[:70] + "..."
                    print(f"    - {text}")


"""
이미지 없는 추천 예외처리 샘플 60개 생성 중...
생성 완료: 60개

=== 샘플 분석 ===
총 샘플 수: 60개

[예외 타입 분포]
  no_info: 20개 (33.3%)
  gender_only: 20개 (33.3%)
  face_shape_only: 20개 (33.3%)

[복잡도 분포]
  simple: 18개 (30.0%)
  medium: 24개 (40.0%)
  complex: 18개 (30.0%)

[타입별 복잡도 분포]
  no_info:
    - simple: 6개
    - medium: 8개
    - complex: 6개
  gender_only:
    - simple: 6개
    - medium: 8개
    - complex: 6개
  face_shape_only:
    - simple: 6개
    - medium: 8개
    - complex: 6개

[질의 길이]
  최소: 12자
  최대: 142자
  평균: 58.7자

저장 완료: samples/no_image_exception.jsonl (60개 샘플)

=== 타입별 & 복잡도별 샘플 미리보기 ===

============================================================
[NO_INFO]
응답: 성별과 얼굴형 또는 퍼스널컬러를 알려주셔야 헤어스타일 추천이 가능합니다...

  [simple]
    - 헤어스타일 추천해줘
    - 어떤 머리가 좋을까?

  [medium]
    - 다음주 소개팅인데 호감가는 스타일로 추천해줘
    - 여름이라 시원해보이는 헤어스타일 하고싶은데 뭐가 좋을까?

  [complex]
    - 이제 여름이 되기도 했고 시원하고 쾌적해보이는 머리스타일을 하고싶은데 어떤게 좋을...
    - 다음주에 면접이 있어서 깔끔하면서도 너무 딱딱해보이지 않는 스타일을 찾고있어. 직...

============================================================
[GENDER_ONLY]
응답: 얼굴형을 알려주셔야 헤어스타일 추천이 가능합니다...

  [simple]
    - 여자인데 머리 추천해줘~
    - 남자인데 어떤 헤어스타일이 좋을까요?

  [medium]
    - 여자인데 직장인이라 너무 튀지않는 스타일 추천해줘
    - 남자고 곱슬머리인데 관리 쉬운 스타일 뭐가 있어?

  [complex]
    - 저 여자인데요, 결혼식 하객으로 가야하는데 너무 화려하지도 않고 그렇다고 너무 평범...
    - 남자인데 곱슬머리라서 항상 머리가 부스스한데 이런 모발에도 잘 어울리면서 관리하기...

============================================================
[FACE_SHAPE_ONLY]
응답: 성별을 알려주셔야 헤어스타일 추천이 가능합니다...

  [simple]
    - 둥근 얼굴인데 어울리는 머리 추천해줘
    - 사각턱인데 뭐가 좋아?

  [medium]
    - 긴 얼굴형인데 얼굴이 짧아보이는 스타일 있을까?
    - 계란형 얼굴인데 요즘 유행하는 스타일로 추천해줘

  [complex]
    - 둥근 얼굴인데 다음주 소개팅이 있어서 얼굴이 좀 갸름해보이면서도 부드러운 인상을 줄...
    - 긴 얼굴형이라 항상 머리 고민이 많은데, 얼굴이 짧아보이면서도 트렌디한 스타일 없을...
"""