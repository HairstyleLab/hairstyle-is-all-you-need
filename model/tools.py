import os
import base64
import tempfile
import stone
import base64
import json
from langchain_classic.agents import load_tools
from langchain_tavily import TavilySearch
# from model.utils import generate_hairstyle
from model.utils import get_face_shape_and_gender, classify_personal_color,get_all_hairstyle,get_hairstyle_list,get_pc_list,get_seasonal_hairstyle_list,get_all_haircolor
from rag.retrieval import load_retriever

def skin_tone_choice(result):
    dominant_result = tuple(int(result['faces'][0]['dominant_colors'][0]['color'].lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    nondominant_result = tuple(int(result['faces'][0]['dominant_colors'][1]['color'].lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    d1,_,_ = dominant_result
    n1,_,_ = nondominant_result
    if d1 > n1:
        return dominant_result
    else:
        return nondominant_result

def hairstyle_recommendation(model, image_base64, query:str, season:str):
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
        face_shape, gender = get_face_shape_and_gender(model, temp_path)


        ## 여기부터 수정버전
        with open("config/hairstyle_list.json", "r", encoding="utf-8") as f:
            data = json.load(f)

        hairstyle_scores = []
        haircolor_scores = []
        all_hairstyle_list = data['전체 헤어스타일'][gender]
        all_haircolor_list = data['전체 헤어스타일']['컬러']
        faceshape_hairstyle_list = data['얼굴형'][gender+face_shape]
        haircolor_list = data['퍼스널컬러'][personal_color]

        if season is not None:
            seasonal_hairstyle_list = data['계절'][gender+season]

        _, vectorstore = load_retriever("rag/db",k=450)

        # 얼굴형별 어울리는 헤어스타일 리스트에서 하나씩 뽑아서 유사도 평균 낸 다음 가중치 더해서 최종 score 저장
        for hairstyle in all_hairstyle_list:
            hairstyle_results = vectorstore.similarity_search_with_relevance_scores(query=query,k=450,filter={"details":hairstyle, 'gender':gender})
            avg_score = sum(score for _, score in hairstyle_results) / len(hairstyle_results)
            face_score = 1 if hairstyle in faceshape_hairstyle_list else 0 
            if season is not None:
                season_score = 1 if hairstyle in seasonal_hairstyle_list else 0 
            else:
                season_score = 0
            hairstyle_scores.append([ hairstyle , 0.4 * avg_score + 0.3 * face_score + 0.3 * season_score ]) 
        
        # 퍼컬별 어울리는 헤어컬러 리스트에서 하나씩 뽑아서 유사도 평균 낸 다음 가중치 더해서 최종 score 저장
        for haircolor in all_haircolor_list:
            haircolor_results = vectorstore.similarity_search_with_relevance_scores(query=query,k=450,filter={"details":haircolor, 'color':'true'})
            color_avg_score = sum(score for _, score in haircolor_results) / len(haircolor_results)
            pc_score = 1 if haircolor in haircolor_list else 0
            haircolor_scores.append( [haircolor, 0.6 * color_avg_score + 0.4 * pc_score])

        # 각각 scores 딕셔너리 정렬
        hairstyles = sorted(hairstyle_scores,key=lambda x:x[1], reverse=True)[:3]
        haircolors = sorted(haircolor_scores,key=lambda x:x[1], reverse=True)[:3]

        # 각 헤어스타일과 헤어컬러에 해당하는 doc 서치해서 저장
        hairstyle_docs = {}
        for hairstyle, _ in hairstyles:
            hair_result,_ = vectorstore.similarity_search_with_relevance_scores(query=query, k=1, filter={'details':hairstyle,'gender':gender})
            hairstyle_docs[hairstyle] = hair_result.page_content

        haircolor_docs = {}
        for haircolor, _ in haircolors:
            color_result,_ = vectorstore.similarity_search_with_relevance_scores(query=query, k=1, filter={'details':haircolor,'color':'true'})
            haircolor_docs[haircolor] = color_result.page_content
        
        return hairstyle_docs, haircolor_docs
        
    finally:
        os.unlink(temp_path)

def hairstyle_generation(model, face_img, shape_img, color_img):
    result = generate_hairstyle(model, face_img, shape_img, color_img)
    return result

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