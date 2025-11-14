import json
import re
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

def summary(old_path, inst, keyword, summary_path):
    with open(old_path, 'r', encoding='utf-8') as f:
        tmp = json.load(f)

    er = []

    new_cont = []
    for i, doc in enumerate(tmp):
        if doc['keyword'] != keyword:
            continue
        
        content = doc['content']
        print(len(content))
        print(content)

        try:
            json_tool = {}
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {'role': 'system', 'content': inst},
                    {'role': 'user', 'content': content},
                ],
                temperature=0,
            )

            content = response.choices[0].message.content
            content = re.sub(r"[^0-9a-zA-Z가-힣\s\(\)\[\]\.: \-]+|\n", " ", content)
            content = re.sub(r"\s+", " ", content).strip()

            # print('gpt response:')
            # print(response)
            # print('\n' + '-' * 50 + '\n')

            json_tool['content'] = content
            json_tool['title'] = doc['title']
            json_tool['link'] = doc['link']
            json_tool['keyword'] = doc['keyword']

            new_cont.append(json_tool)

        except Exception as e:
            print(f"{i}번째 처리 중 오류 발생: {e}")
            er.append(i)

    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(new_cont, f, ensure_ascii=False, indent=4)

if __name__=="__main__":
    path = 'summary_content.json'
    inst = '''
당신은 미용 게시물을 요약해주는 노련한 블로거입니다.

- 입력은 문자열으로 주어집니다.
- 입력에서 얼굴형별 헤어스타일 추천에 관한 정보만을 수집합니다.
- 수집한 정보만 사용해 글을 100자 내외로 요약합니다.
- 입력을 토대로 요약을 진행하고, 입력에 나와있지 않은 내용은 언급하면 안됩니다.
- 출력은 반드시 특수문자, 기호 없이 문자와 띄어쓰기, 기본 문장부호로만 이루어져야 합니다.
'''
    summary(path, inst, '얼굴형별 헤어스타일', 'summary_cat_hair_face.json')
