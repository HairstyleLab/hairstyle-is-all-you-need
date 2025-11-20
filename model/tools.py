import os
import json
import base64
import tempfile
import stone
import base64
import json
from langchain_classic.agents import load_tools
from langchain_tavily import TavilySearch
# from model.utils import generate_hairstyle
from model.utils import get_face_shape_and_gender, classify_personal_color,get_faceshape
from model.model_load import load_embedding_model, load_reranker_model
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

def hairstyle_recommendation(model, image_base64, query:str, season=None):
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
        
        embeddings = load_embedding_model("dragonkue/snowflake-arctic-embed-l-v2.0-ko", device="cpu")
        _, vectorstore = load_retriever("rag/db/all_merge_hf", embeddings=embeddings)
        # _, vectorstore = load_retriever("rag/db/all_split_merge_hf",k=450)


        # 얼굴형별 어울리는 헤어스타일 리스트에서 하나씩 뽑아서 유사도 평균 낸 다음 가중치 더해서 최종 score 저장
        for hairstyle in all_hairstyle_list:
            hairstyle_results = vectorstore.similarity_search_with_score(query=query,k=450,fetch_k=450,filter={'gender':gender,'details':hairstyle})
            try:
                avg_score = sum(score for _, score in hairstyle_results) / len(hairstyle_results)
                face_score = 1 if hairstyle in faceshape_hairstyle_list else 0 
                if season is not None:
                    season_score = 1 if hairstyle in seasonal_hairstyle_list else 0 
                else:
                    season_score = 0
                hairstyle_scores.append([ hairstyle , 0.4 * avg_score + 0.3 * face_score + 0.3 * season_score ]) 
            except:
                continue
        
        # 퍼컬별 어울리는 헤어컬러 리스트에서 하나씩 뽑아서 유사도 평균 낸 다음 가중치 더해서 최종 score 저장
        for haircolor in all_haircolor_list:
            haircolor_results = vectorstore.similarity_search_with_score(query=query,k=450,fetch_k=450,filter={"details":haircolor})
            try:
                color_avg_score = sum(score for _, score in haircolor_results) / len(haircolor_results)
                pc_score = 1 if haircolor in haircolor_list else 0
                haircolor_scores.append( [haircolor, 0.6 * color_avg_score + 0.4 * pc_score])
            except:
                continue

        # 각각 scores 리스트 정렬
        hairstyles = sorted(hairstyle_scores,key=lambda x:x[1], reverse=True)[:3]
        haircolors = sorted(haircolor_scores,key=lambda x:x[1], reverse=True)[:3]
        print(hairstyles,haircolors)

        # 각 헤어스타일과 헤어컬러에 해당하는 doc 서치해서 저장
        hairstyle_docs = {}
        for hairstyle, _ in hairstyles:
            hair_result = vectorstore.similarity_search_with_score(query=query, k=1, fetch_k=450, filter={'details':hairstyle,'gender':gender})
            hairstyle_docs[hairstyle] = [doc.page_content for doc,_ in hair_result]

        haircolor_docs = {}
        for haircolor, _ in haircolors:
            color_result = vectorstore.similarity_search_with_score(query=query, k=1, fetch_k=450, filter={'details':haircolor})
            haircolor_docs[haircolor] = [doc.page_content for doc,_ in color_result]

        faceshape_docs = {}
        korean_faceshape = get_faceshape(face_shape)
        faceshape_result = vectorstore.similarity_search_with_score(query=query,k=2,fetch_k=450,filter={'details':korean_faceshape,'gender':gender})
        faceshape_docs[korean_faceshape] = [doc.page_content for doc,_ in faceshape_result]
        
        
        return faceshape_docs, personal_color, hairstyle_docs, haircolor_docs
        
    finally:
        os.unlink(temp_path)

# def hairstyle_generation(model, face_img, shape_img, color_img):
#     result = generate_hairstyle(model, face_img, shape_img, color_img)
#     return result

def hairstyle_generation(image_base64, hairstyle=None, haircolor=None, client=None):
    if image_base64.startswith('data:image'):
        image_data = base64.b64decode(image_base64.split(',')[1])
    else:
        image_data = base64.b64decode(image_base64)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
        temp_file.write(image_data)
        temp_path = temp_file.name

    with open('config/reference.json', 'r', encoding='utf-8') as f:
        reference = json.load(f)

    image = None
    hairstyle_path = None
    haircolor_path = None
    hairstyle_dict = reference.get("헤어스타일", {})

    if hairstyle:
        for gender in hairstyle_dict.values():
            for category in gender.values():
                if hairstyle in category:
                    hairstyle_path = category[hairstyle]
                    break
            if hairstyle_path:
                break
    if haircolor:
        color_dict = reference.get("컬러", {})
        haircolor_path = color_dict.get(haircolor, None)

    if hairstyle_path and haircolor_path:
        prompt = """첫번째 이미지의 사람 헤어스타일을 두번째 이미지의 사람 헤어스타일로 바꾸고 세번째 이미지의 사람 헤어컬러를 적용해줘.
                    이미지를 생성할때 첫번째 이미지의 사람 그대로 생성하되 헤어스타일과 헤어컬러만 바뀌어야 해."""
        image = generate_image(client, prompt, image_path=temp_path, shape_path=hairstyle_path, color_path=haircolor_path)
    elif hairstyle_path and haircolor_path is None:
        prompt = """첫번째 이미지의 사람 헤어스타일을 두번째 이미지의 사람 헤어스타일로 적용해주고 헤어컬러는 기존 그대로 유지해줘.
                    이미지를 생성할때 첫번째 이미지의 사람 그대로 생성하되 헤어스타일만 바뀌어야 해."""
        image = generate_image(client, prompt, image_path=temp_path, shape_path=hairstyle_path)
    elif haircolor_path and hairstyle_path is None:
        prompt = """첫번째 이미지의 사람 헤어컬러만 두번째 이미지의 사람 컬러로 바꿔줘.
                    이미지를 생성할때 첫번째 이미지의 사람 그대로 생성하되 헤어컬러만 바뀌어야 해."""
        image = generate_image(client, prompt, image_path=temp_path, color_path=haircolor_path)

    folder_path = "C:\\Users\\Playdata\\Desktop\\hairstyle-is-all-you-need\\results"
    path = len([file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))])
    with open(f"results/{path}.jpg", "wb") as f:
        f.write(image)

    return "이미지 생성 완료."

def safe_open(path):
    if path and os.path.exists(path):
        return open(path, "rb")
    return None

def generate_image(client, prompt, image_path, shape_path=None, color_path=None):
    image_inputs = [
        safe_open(image_path),
        safe_open(shape_path),
        safe_open(color_path),
    ]
    image_inputs = [img for img in image_inputs if img is not None]

    result = client.images.edit(
        model="gpt-image-1",
        image=image_inputs,
        prompt=prompt
    )

    image_base64 = result.data[0].b64_json
    image_bytes = base64.b64decode(image_base64)

    return image_bytes

def web_search(query: str)->str:
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

def rag_search(face_shape: str|None=None, season: str|None=None, tone: str|None=None):
    embeddings = load_embedding_model("dragonkue/snowflake-arctic-embed-l-v2.0-ko", device="cpu")
    retriever, _ = load_retriever("rag/db/all_merge_hf", embeddings=embeddings, k=10)

    res = []
    if face_shape:
        res += retriever.invoke(face_shape, filter={'category': 'face'}, k=3)
    if season:
        res += retriever.invoke(season, filter={'category': 'season'}, k=3)
    if tone:
        res += retriever.invoke(tone, filter={'category': 'skintone'}, k=3)
    
    return res

def get_tool_list(*args):
    tools = load_tools(['dalle-image-generator'])
    tools.extend(args)
    return tools