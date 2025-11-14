import os
import re
import requests
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
from model.agent_openai import build_agent
from model.utils import load_identiface
import base64
from langchain_core.messages import HumanMessage
from PIL import Image
# from model.utils import load_hairfastgan, generate_hairstyle
from model.utils import load_identiface, get_face_shape_and_gender

file_path = "images/face2.jpg"
face_img = "images/ky.jpg"
shape_img = "images/ew.jpg"
color_img = "images/gd.jpg"

# model = load_hairfastgan()
# result = generate_hairstyle(model, face_img, shape_img, color_img)

model = load_identiface()
predicted_shape, shape_probs = get_face_shape_and_gender(model, file_path)

load_dotenv()

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

agent = build_agent(model)
encoded_image = encode_image_from_file(file_path)

def make_human_message(input_text,session_id,file_path=None):
    if not file_path:
        response = agent.invoke(
            {"input": input_text},
            config={"configurable": {"session_id": session_id}}
        )
        print(response['output'])
        
    else:
        response = agent.invoke(
            {"input": [HumanMessage(content=[
                {"type": "text", "text": input_text},
                {"type": "image_url", "image_url": {"url": encoded_image}}
            ])]},
            config={"configurable": {"session_id": session_id}}
        )
        print(response['output'])
        extract_and_display_image(response)

print(make_human_message("이 머리에 어울리는 헤어스타일 추천해줘", session_id="test_session1", file_path=file_path))
print(make_human_message("이 얼굴에 히피펌 적용한 사진 줘볼래?", session_id="test_session2", file_path=file_path))
print("\n\n테스트 완료!")




