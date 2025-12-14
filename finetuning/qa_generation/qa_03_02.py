import json
import base64
import random
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()


def generate_image_recommendation_exception_queries(num_samples: int = 10) -> list:
    """이미지 있는 추천 예외처리 질의 생성"""
    
    prompt = f"""
헤어스타일 추천 챗봇의 학습 데이터를 생성해주세요.

[시나리오]
사용자가 사진을 업로드하고 헤어스타일 추천을 요청하는 경우입니다.
하지만 이미지에 문제가 있어서 예외처리가 필요한 상황입니다.

[예외 케이스 2가지]

1. 얼굴 없는 이미지 (풍경, 음식, 동물, 물체 등)
   → 응답: "얼굴이 포함된 이미지를 첨부하셔야 헤어스타일 추천이 가능합니다🥲 확인 후 다른 사진을 업로드해주세요."

2. 얼굴 2명 이상 이미지 (단체사진, 커플사진 등)
   → 응답: "이 이미지에는 2명 이상의 얼굴이 포함되어 있습니다🥲 한 명만 나온 이미지를 업로드해주세요."

[중요: 질의 복잡도 분포]
- 단순 질의 (30%): "이 사진으로 헤어 추천해줘", "어떤 머리가 어울려?"
- 중간 복잡도 (40%): 상황/맥락 포함, 조건 1-2개
- 복잡한 질의 (30%): 여러 조건, 상세한 맥락, 긴 문장


[복잡한 질의 예시 - 반드시 이런 스타일 포함]
- "이제 여름이 되기도 했고 시원하고 쾌적해보이는 머리스타일을 하고싶은데 어떤게 좋을까? 그러면서도 내 얼굴에 어울리는 헤어스타일이면 좋겠고 색은 톤다운된 컬러로 너무 튀지는 않았으면 좋겠어"
- "다음주에 면접이 있어서 깔끔하면서도 너무 딱딱해보이지 않는 스타일을 찾고있어. 직장인으로서 무난하면서도 세련된 느낌이었으면 좋겠는데 내 얼굴형에는 뭐가 맞을까?"
- "요즘 머리가 너무 상해서 펌은 하고싶은데 손상이 덜한 걸로 하고싶거든? 근데 또 볼륨은 있었으면 좋겠고... 내 얼굴이랑 매치되는 스타일 좀 추천해줄 수 있어?"
- "결혼식 하객으로 가야하는데 너무 화려하지도 않고 그렇다고 너무 평범하지도 않은 헤어스타일 있을까? 드레스코드가 세미포멀이라 그거에 맞게 우아한 느낌으로 하고싶어"
- "저 곱슬머리라서 항상 머리가 부스스한데 이런 모발에도 잘 어울리면서 관리하기 쉬운 스타일이 뭐가 있을까요? 아침에 손질 시간 많이 못 쓰거든요"

[맥락/상황 키워드 - 적극 활용]
- 계절: 여름이라 시원하게, 겨울이라 따뜻해보이게, 환절기라, 봄맞이로
- 이벤트: 면접, 소개팅, 결혼식, 졸업식, 입학식, 여행, 데이트, 동창회, 회사워크샵
- 고민: 머리가 상해서, 탈모가 있어서, 얼굴이 커서, 이마가 넓어서, 광대가 나와서
- 조건: 관리가 쉬운, 손질 안해도 되는, 오래 유지되는, 자연스러운, 세련된
- 직업/상황: 직장인이라, 학생이라, 취준생이라, 아이 엄마라, 운동을 많이 해서
- 스타일: 청순한, 시크한, 귀여운, 성숙한, 지적인, 편안한, 단정한, 트렌디한
- 복합 조건: A하면서도 B한, A인데 B는 싫고, A랑 B 둘 다 가능한

[질의에 포함될 수 있는 요청 유형]
- 스타일 추천: "어떤 머리가 어울려?", "뭐가 좋을까?"
- 컬러 추천: "어떤 색이 맞을까?", "염색은 뭐로?"
- 종합 추천: "스타일이랑 색 둘 다", "전체적으로 바꾸고 싶어"
- 조언 요청: "어떻게 하면 좋을지", "고민인데 도와줘"

[다양한 말투]
- 반말 친근체: "~해줘", "~할까?", "~좋겠어", "~싶거든"
- 존댓말: "~해주세요", "~할까요?", "~좋겠습니다"
- 이모지 사용: 😊, 🤔, ㅠㅠ, ㅎㅎ, ~!, ...
- 구어체: "근데", "그냥", "좀", "막", "되게", "엄청"

[생성 규칙]
각 케이스별로 다양한 질의를 생성해주세요:
- 케이스1 (얼굴 없음): {num_samples // 2}개
- 케이스2 (2명 이상): {num_samples // 2}개

[출력 형식]
JSON 배열로 출력. 각 항목:
{{
  "type": "no_face" | "multi_face",
  "user": "사용자 질의",
  "complexity": "simple" | "medium" | "complex"
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


EXCEPTION_RESPONSES = {
    "no_face": "얼굴이 포함된 이미지를 첨부하셔야 헤어스타일 추천이 가능합니다🥲 확인 후 다른 사진을 업로드해주세요.",
    "multi_face": "이 이미지에는 2명 이상의 얼굴이 포함되어 있습니다🥲 한 명만 나온 이미지를 업로드해주세요."
}


def convert_to_training_format(samples: list) -> list:
    training_data = []
    
    for i, sample in enumerate(samples):
        
        response_text = EXCEPTION_RESPONSES[sample["type"]]
        
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
                    "content": response_text
                }
            ],
            "image_type": sample["type"]
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
    
    type_counts = {"no_face": 0, "multi_face": 0}
    complexity_counts = {"simple": 0, "medium": 0, "complex": 0}
    
    for sample in samples:
        type_counts[sample["type"]] += 1
        complexity_counts[sample.get("complexity", "medium")] += 1
    
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
    
    # 길이 분석
    lengths = [len(s["user"]) for s in samples]
    print(f"\n[질의 길이]")
    print(f"  최소: {min(lengths)}자")
    print(f"  최대: {max(lengths)}자")
    print(f"  평균: {sum(lengths)/len(lengths):.1f}자")


def get_data(
    num_samples: int = 10,
    output_file: str = "finetuning/samples/qa_03_02.jsonl"
):
    """메인 함수: 텍스트 질의-응답 생성 (이미지는 build_dataset.py에서 매칭)"""

    print(f"예외처리 샘플 {num_samples}개 생성 중...")
    samples = generate_image_recommendation_exception_queries(num_samples)
    print(f"생성 완료: {len(samples)}개")

    analyze_samples(samples)

    training_data = convert_to_training_format(samples)

    save_to_jsonl(training_data, output_file)

    return training_data, samples


if __name__ == "__main__":
    training_data, raw_samples = get_data(
        num_samples=100,
        output_file="finetuning/samples/qa_03_02.jsonl"
    )
    
    print("\n=== 복잡도별 샘플 미리보기 ===")
    
    for complexity in ["simple", "medium", "complex"]:
        filtered = [s for s in raw_samples if s.get("complexity") == complexity]
        print(f"\n[{complexity.upper()}]")
        for sample in filtered[:3]:
            print(f"  - [{sample['type']}] {sample['user'][:80]}{'...' if len(sample['user']) > 80 else ''}")


"""
얼굴 없는 이미지 로드 중: images/no_face
로드된 이미지: 5개
다중 얼굴 이미지 로드 중: images/multi_face
로드된 이미지: 5개

예외처리 샘플 60개 생성 중...
생성 완료: 60개

=== 샘플 분석 ===
총 샘플 수: 60개

[예외 타입 분포]
  no_face: 30개 (50.0%)
  multi_face: 30개 (50.0%)

[복잡도 분포]
  simple: 18개 (30.0%)
  medium: 24개 (40.0%)
  complex: 18개 (30.0%)

[질의 길이]
  최소: 15자
  최대: 156자
  평균: 62.3자

저장 완료: samples/exception_handling.jsonl (60개 샘플)

=== 복잡도별 샘플 미리보기 ===

[SIMPLE]
  - [no_face] 이 사진으로 어떤 헤어가 어울릴까?
  - [multi_face] 머리 스타일 추천해주세요
  - [no_face] 나한테 맞는 헤어 뭐야?

[MEDIUM]
  - [multi_face] 다음주 소개팅인데 깔끔하면서 호감가는 스타일로 추천해줘
  - [no_face] 여름이라 시원해보이는 헤어스타일 하고싶은데 뭐가 좋을까요?
  - [multi_face] 직장인인데 너무 튀지않으면서 세련된 머리 추천해줄 수 있어?

[COMPLEX]
  - [no_face] 이제 여름이 되기도 했고 시원하고 쾌적해보이는 머리스타일을 하고싶은데 어떤게 ...
  - [multi_face] 다음주에 면접이 있어서 깔끔하면서도 너무 딱딱해보이지 않는 스타일을 찾고있어...
  - [no_face] 저 곱슬머리라서 항상 머리가 부스스한데 이런 모발에도 잘 어울리면서 관리하기 쉬...
"""