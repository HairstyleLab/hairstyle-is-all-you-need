import os
import re
import requests
from io import BytesIO
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv
from model.agent_openai import build_agent
# from model.agent_hf import build_agent
from model.utils import load_identiface
import base64
from langchain_core.messages import HumanMessage
from PIL import Image
# from model.utils import load_hairfastgan, generate_hairstyle
from model.utils import load_identiface, get_face_shape_and_gender
from model.model_load import load_embedding_model, load_safmn_model, load_face_cropper, load_3d_models
from rag.retrieval import load_retriever
import warnings
warnings.filterwarnings("ignore")

file_path = "images/dw2.jpg"  # 테스트용 이미지 파일 경로
load_dotenv()

print("=" * 50)
print("테스트 환경 모델 로딩 시작...")
print("=" * 50)

# 1. IdentiFace 모델 로드
print("\n[1/6] IdentiFace 모델 로드 중...")
model = load_identiface()

# 2. OpenAI 클라이언트
client = OpenAI()

# 3. 임베딩 모델 및 벡터스토어 로드
print("\n[2/6] 임베딩 모델 및 벡터스토어 로드 중...")
embeddings = load_embedding_model("dragonkue/snowflake-arctic-embed-l-v2.0-ko", device="cuda")
_, vectorstore = load_retriever("rag/db/styles_added_hf", embeddings)

# 4. SAFMN 초해상도 모델 로드
print("\n[3/6] SAFMN 초해상도 모델 로드 중...")
safmn_model = load_safmn_model(device="cuda")

# 5. FaceCropper 로드
print("\n[4/6] FaceCropper 로드 중...")
face_cropper = load_face_cropper(crop_size=256)

# 6. 3D 재구성 모델들 로드
print("\n[5/6] 3D 재구성 모델들 로드 중...")
models_3d = load_3d_models(device="cuda")

# 7. Agent 생성
print("\n[6/6] Agent 생성 중...")
agent = build_agent(model, client, vectorstore, safmn_model, face_cropper, models_3d)

print("\n" + "=" * 50)
print("모든 모델 로딩 완료!")
print("=" * 50 + "\n")

def extract_and_display_image(response,image_path=os.path.abspath("images")):
    output = response.get('output', '')
    url_pattern = r'https://oaidalleapiprodscus[^\s\)]+\.png[^\s\)]*'
    urls = re.findall(url_pattern, output)
    
    if urls:
        for i, url in enumerate(urls):
            img_response = requests.get(url)
            img = Image.open(BytesIO(img_response.content))
            filename = f"generated_hairstyle_{i+1}.png"
            img.save(os.path.join(image_path, filename))
            img.show()

def encode_image_from_file(file_path):
    with open(file_path, "rb") as image_file:
        image_content = image_file.read()
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext in [".jpg", ".jpeg"]:
            mime_type = "image/jpeg"
        elif file_ext == ".png":
            mime_type = "image/png"
        else:
            mime_type = "image/unknown"
        return f"data:{mime_type};base64,{base64.b64encode(image_content).decode('utf-8')}"

def make_human_message(input_text,session_id,file_path=None):
    agent.gen_flag=False

    if not file_path:
        print('out image')
        print(input_text)
        response = agent.invoke(
            {"input": [HumanMessage(content=[
                {"type": "text", "text": input_text}
            ])]},
            config={"configurable": {"session_id": session_id}}
        )
        print(response['output'])
        
    else:
        encoded_image = encode_image_from_file(file_path)
        print('in image ', file_path)
        response = agent.invoke(
            {"input": [HumanMessage(content=[
                {"type": "text", "text": input_text},
                {"type": "image_url", "image_url": {"url": encoded_image}}
            ])]},
            config={"configurable": {"session_id": session_id}}
        )
        print(response['output'])
    
    return response, agent.gen_flag

q1 = "이 얼굴에 히피펌 헤어스타일에 애쉬그레이 컬러를 적용한 이미지를 생성해줄래?" # (이미지 생성시, 헤어스타일, 헤어컬러 둘 다 명시한 경우)
q2 = "이 얼굴에 히피펌 헤어스타일을 적용한 이미지를 생성해줄래?" # (이미지 생성시, 헤어스타일만 명시한 경우)
q3 = "이 얼굴에 애쉬그레이를 적용한 이미지를 생성해줄래?" # (이미지 생성시, 헤어컬러만 명시한 경우)
q4 = "이 얼굴에 히피펌 헤어스타일에 애쉬그레이 컬러를 적용한 이미지를 생성해줄래?" # (이미지 생성 시 이미지 업로드 안한 경우)
q5 = "이 얼굴에 마땅한 헤어스타일과 컬러를 적용한 이미지를 생성해줄래?" # (이미지 생성 시 헤어스타일과 헤어컬러를 언급 안한 경우)
q6 = "이 얼굴에 히피 펆 헤어스타일에 애시 그래 컬러를 적용한 이미지를 생성해줄래?" # (이미지 생성 시 헤어스타일, 헤어컬러를 살짝 틀린 문자열로 표현한 경우)
q7 = "이 얼굴에 마이쮸펌 헤어스타일에 칙칙한 초코칩 컬러를 적용한 이미지를 생성해줄래?" # (이미지 생성 시 등록된 헤어스타일, 헤어컬러가 아닌 옵션으로 표현한 경우)
q8 = "남자 둥근형 얼굴에 어울리는 헤어스타일을 추천해줘."

# print(make_human_message("이 머리에 어울리는 헤어스타일 추천해줘", session_id="test_session1", file_path=file_path))
print(make_human_message(q8, session_id="test_session2", file_path=file_path))
# print("\n\n테스트 완료!")

# while True:
#     query = input("질문: ") # 사용자 질문 입력
#     if query.lower() in ["exit", "quit"]:
#         print("챗봇을 종료합니다.")
#         break
#     file_check = input("이미지를 업로드할지 안할지 입력하시오.(y/n): ") # 사용자 질문 입력

#     if file_check == "y":
#         make_human_message(query, session_id="test_session2", file_path=file_path)
#     else:
#         make_human_message(query, session_id="test_session2", file_path=None)

# QA 캐시 테스트 (이미지 없이)
# print("=== QA 캐시 테스트 시작 ===\n")

# test_queries = [
#     "여름에 사각형 얼굴의 여자가 하기 좋은 산뜻한 단발 추천해줘",
#     "남자 계란 얼굴에 어울리는 시원한 헤어스타일 알려줘",
# ]

# for i, query in enumerate(test_queries, 1):
#     print(f"\n[테스트 {i}] {query}")
#     make_human_message(query, session_id="qa_cache_test", file_path=None)
#     print("-" * 80)

# print("\n=== QA 캐시 테스트 완료 ===")

# # 기존 이미지 생성 테스트
# file_list = [["images/j_1.jpg"], ["images/j_2.jpg"], ["images/j_3.jpg"], ["images/j_4.jpg"]]
# for i in range(4):
#     query = "이 얼굴에 히피펌이랑 애쉬그레이 컬러를 적용한 이미지를 생성해줄래?"
#     make_human_message(query, session_id="test_session2", file_path=file_list[i])
