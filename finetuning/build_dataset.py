import json

def get_images():
    # 얼굴1명 / 얼굴 다수 / 얼굴 없는 이미지 다수 불러오기
    # 각 카테고리별로 저장된 이미지들 폴더 경로가 기록된 json 파일(제작 필요)로부터 로드하기
    return

def get_data():
    # OpenAI API 기반 데이터 추출 -> jsonl 파일 저장 로직
    # get_images 함수를 통해 다양한 질의에 대한 다양한 이미지 경우의 수 삽입
    return 

def build_training_data():
    with open("config/tools.json", "r", encoding="utf-8") as f:
        tools = json.load(f)
    
    with open("config/system_prompt.txt", "r", encoding="utf-8") as f:
        system_prompt = f.read().strip()
    
    with open("samples.jsonl", "r", encoding="utf-8") as f_in, \
         open("training_data.jsonl", "w", encoding="utf-8") as f_out:
        
        for line in f_in:
            sample = json.loads(line.strip())
            
            if sample["messages"][0]["role"] == "system":
                sample["messages"][0]["content"] = system_prompt
            else:
                sample["messages"].insert(0, {"role": "system", "content": system_prompt})
            
            sample["tools"] = tools
            
            f_out.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    print("완료!")