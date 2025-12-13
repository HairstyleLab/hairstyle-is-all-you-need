import json
import random
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()


def load_preprocessed_images(json_file: str) -> list:
    """미리 인코딩된 base64 이미지 JSON 파일 로드 (파일이 없으면 빈 리스트 반환)"""
    p = Path(json_file)
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def merge_jsonl_to_jsonl(input_dir: str, output_file: str):
    input_path = Path(input_dir)

    with open(output_file, "w", encoding="utf-8") as f_out:
        for jsonl_file in sorted(input_path.glob("*.jsonl")):
            print(f"Processing {jsonl_file}...")
            with open(jsonl_file, "r", encoding="utf-8") as f_in:
                for line in f_in:
                    line = line.strip()
                    if not line:
                        continue
                    f_out.write(line + "\n")

    print(f"완료! JSONL 파일 생성 → {output_file}")


def build_training_data(
    input_jsonl="training_data.jsonl",
    output_jsonl="training_data_final.jsonl",
    images_map=None,
    tools_config_path="finetuning/config/tools_config.json",
    system_prompt_path="finetuning/config/system_prompt.txt"
):
    """
    images_map 예시:
    {
        "no_face": "finetuning/images/no_face.json",
        "multi_face": "finetuning/images/multi_face.json",
        "normal": "finetuning/images/normal_face.json"
    }

    이 함수는:
    1. 각 qa 파일에서 생성된 중간 JSONL 파일들을 읽음
    2. 각 샘플의 image_type에 맞는 이미지를 랜덤으로 매칭
    3. system prompt와 tools 설정 추가
    4. 최종 JSONL 파일로 저장
    """

    images_map = images_map or {}

    tools = []
    if Path(tools_config_path).exists():
        with open(tools_config_path, "r", encoding="utf-8") as f:
            tools = json.load(f)
        print(f"Loaded {len(tools)} tools from {tools_config_path}")

    system_prompt = ""
    if Path(system_prompt_path).exists():
        with open(system_prompt_path, "r", encoding="utf-8") as f:
            system_prompt = f.read().strip()
        print(f"Loaded system prompt from {system_prompt_path}")

    images_by_type = {}
    for k, path in images_map.items():
        imgs = load_preprocessed_images(path)
        images_by_type[k] = imgs
        print(f"Loaded {len(imgs)} images for type '{k}' from {path}")

    out_path = Path(output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_jsonl, "r", encoding="utf-8") as f_in, \
         open(output_jsonl, "w", encoding="utf-8") as f_out:

        for idx, line in enumerate(f_in):
            sample = json.loads(line.strip())

            if sample.get("messages") and len(sample["messages"]) > 0 and sample["messages"][0].get("role") == "system":
                sample["messages"][0]["content"] = system_prompt
            else:
                sample.setdefault("messages", [])
                sample["messages"].insert(0, {"role": "system", "content": system_prompt})

            image_type = sample.get("image_type")

            for msg in sample["messages"]:
                if msg.get("role") != "user":
                    continue

                if isinstance(msg.get("content"), list):

                    for content_item in msg["content"]:
                        if content_item.get("type") == "image_url":
                            cur_url = content_item.get("image_url", {}).get("url")
                            if cur_url not in (None, ""):

                                continue

                            if image_type and image_type in images_by_type and images_by_type[image_type]:
                                chosen = random.choice(images_by_type[image_type])
                                if isinstance(chosen, dict):
                                    base64_url = chosen.get("base64", chosen.get("url", ""))
                                else:
                                    base64_url = chosen

                                content_item["image_url"]["url"] = base64_url
                break

            if any(msg.get("role") == "assistant" and msg.get("tool_calls") for msg in sample.get("messages", [])):
                if tools:
                    sample["tools"] = tools

            f_out.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"\n완료! 최종 데이터셋 저장 → {output_jsonl}")
    print(f"총 {idx+1}개 샘플 처리 완료")


if __name__ == "__main__":
    merge_jsonl_to_jsonl("finetuning/samples", "finetuning/training_data.jsonl")
    build_training_data(
        input_jsonl="finetuning/training_data.jsonl",
        output_jsonl="finetuning/training_data_final.jsonl",
        images_map={
            "no_face": "finetuning/images/no_face.json",
            "multi_face": "finetuning/images/multi_face.json",
            "normal": "finetuning/images/normal_face.json"
        },
        tools_config_path="finetuning/config/tools_config.json",
        system_prompt_path="finetuning/config/system_prompt.txt"
    )
