import os
import json
import base64
import tempfile
import stone
import base64
from langchain_classic.agents import load_tools
from langchain_tavily import TavilySearch
# from model.utils import generate_hairstyle
from model.utils import get_face_shape_and_gender, classify_personal_color

def skin_tone_choice(result):
    dominant_result = tuple(int(result['faces'][0]['dominant_colors'][0]['color'].lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    nondominant_result = tuple(int(result['faces'][0]['dominant_colors'][1]['color'].lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    d1,_,_ = dominant_result
    n1,_,_ = nondominant_result
    if d1 > n1:
        return dominant_result
    else:
        return nondominant_result

def hairstyle_recommendation(model, image_base64):
    if image_base64.startswith('data:image'):
        image_data = base64.b64decode(image_base64.split(',')[1])
    else:
        image_data = base64.b64decode(image_base64)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
        temp_file.write(image_data)
        temp_path = temp_file.name
    
    try:
        result = stone.process(temp_path, image_type='color',return_report_image=False,tone_palette='perla')
        skin_tone = skin_tone_choice(result)
        personal_color = classify_personal_color(skin_tone)
        shape, gender = get_face_shape_and_gender(model, temp_path)
        return f"퍼스널컬러:{personal_color},얼굴형: {shape},성별: {gender}"
    finally:
        os.unlink(temp_path)

# def hairstyle_generation(model, face_img, shape_img, color_img):
#     result = generate_hairstyle(model, face_img, shape_img, color_img)
#     return result

def hairstyle_generation(image_base64, hairstyle, haircolor, client):
    if image_base64.startswith('data:image'):
        image_data = base64.b64decode(image_base64.split(',')[1])
    else:
        image_data = base64.b64decode(image_base64)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
        temp_file.write(image_data)
        temp_path = temp_file.name

    with open('config/reference.json', 'r', encoding='utf-8') as f:
        reference = json.load(f)

    prompt = "첫번째 이미지의 사람 헤어스타일을 두번째 이미지의 사람 헤어스타일로 바꾸고 세번째 이미지의 헤어컬러를 적용해줘"
    hairstyle_path = None
    haircolor_path = None

    hairstyle_dict = reference.get("헤어스타일", {})

    for gender in hairstyle_dict.values():
        for category in gender.values():
            if hairstyle in category:
                hairstyle_path = category[hairstyle]
                break
        if hairstyle_path:
            break

    color_dict = reference.get("컬러", {})
    haircolor_path = color_dict.get(haircolor, None)

    image = generate_image(client, prompt, temp_path, hairstyle_path, haircolor_path)
    with open("result/output.jpg", "wb") as f:
        f.write(image)

    return "이미지 생성 완료."

def generate_image(client, prompt, image_path, shape_path, color_path):

    result = client.images.edit(
        model="gpt-image-1",
        image=[
            open(image_path, "rb"),
            open(shape_path, "rb"),
            open(color_path, "rb"),
        ],
        prompt=prompt
    )

    image_base64 = result.data[0].b64_json
    image_bytes = base64.b64decode(image_base64)

    return image_bytes

def web_search(query:str)->str:
    TAVILY_API_KEY=os.getenv("TAVILY_API_KEY")
    tool = TavilySearch(
        max_results=10,
        topic = 'general',
        tavily_api_key=TAVILY_API_KEY,
        include_answer=True,
        search_depth='basic',
    )
    results = tool.invoke(query)
    final_result = results['answer']
    for content in results['results']:
        final_result += f" {content['content']}"
    return final_result

def get_tool_list(*args):
    tools = load_tools(['dalle-image-generator'])
    tools.extend(args)
    return tools