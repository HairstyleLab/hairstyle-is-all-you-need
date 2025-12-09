import json
from openai import OpenAI

client = OpenAI()


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
