import os
import sys
import json
import base64
import tempfile
import stone
from PIL import Image
import numpy as np
from io import BytesIO
from math import inf
from langchain_classic.agents import load_tools
from langchain_tavily import TavilySearch
# from model.utils import generate_hairstyle
from model.utils import get_face_shape_and_gender, classify_personal_color,get_faceshape, get_weight, face_crop, get_3d
from model.model_load import load_embedding_model
from rag.retrieval import load_retriever
from model.utility.superresolution import get_high_resolution
from model.utility.white_balance import grayworld_white_balance
from model.utility.face_swap import face_swap
from rag.qa_cache import QACache


# embeddings = load_embedding_model("dragonkue/snowflake-arctic-embed-l-v2.0-ko", device="cuda")
embeddings = load_embedding_model("dragonkue/snowflake-arctic-embed-l-v2.0-ko", device="cuda")    
retriever, vectorstore = load_retriever("rag/db/new_hf_1211", embeddings)

def skin_tone_choice(result):
    dominant_result = tuple(int(result['faces'][0]['dominant_colors'][0]['color'].lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    nondominant_result = tuple(int(result['faces'][0]['dominant_colors'][1]['color'].lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    d1,_,_ = dominant_result
    n1,_,_ = nondominant_result
    if d1 > n1:
        return dominant_result
    else:
        return nondominant_result
      

def non_image_recommendation(face_shape=None, gender=None, personal_color=None, season=None, hairstyle_keywords=None, haircolor_keywords=None, hairlength_keywords=None, status_callback=None):
    
    
    if status_callback:
        status_callback("추천 헤어스타일 검색 중...")

    scores = {'hair':[],'color':[]}
    results = {'hair':{},'color':{}}
    result_docs = {}
    weight = {'hair': 0,'color': 0}
    hair_max_score, hair_min_score = -inf, inf
    color_max_score, color_min_score = -inf, inf
    with open("config/hairstyle_list.json", "r", encoding="utf-8") as f:
        hairstyle_data = json.load(f)

    with open("config/hairstyle_length.json", "r", encoding="utf-8") as f:
        hairstyle_length = json.load(f)

    # 성별& 얼굴형 있으면
    if gender is not None and face_shape is not None :
        all_hairstyle_length = hairstyle_length['헤어스타일'][gender]
        faceshape_hairstyle_list = hairstyle_data['얼굴형'][gender+face_shape]
        faceshape_hairlength_list = hairstyle_data['얼굴형별 추천 기장'][gender+face_shape]
        # 얼굴형 특징 데이터 먼저 서치해서 문서에 담기
        korean_faceshape = get_faceshape(face_shape)
        faceshape_result = vectorstore.similarity_search_with_score(query=korean_faceshape,k=2,fetch_k=1000,filter={'details':korean_faceshape,'gender':gender})
        result_docs[korean_faceshape] = [doc.page_content for doc,_ in faceshape_result]
        # 키워드가 있으면 원래 로직대로 수행 ( keywords 검색해서 가중치 반영해서 계산해 정렬 )
        if hairstyle_keywords is not None:
            all_hairstyle_list = []

            if hairlength_keywords:
                for key, val in all_hairstyle_length.items():
                    for sub_key, sub_val in val.items():
                        if hairlength_keywords in sub_val:
                            all_hairstyle_list.append(sub_key)
            else:
                for key, val in all_hairstyle_length.items():
                    for sub_key, sub_val in val.items():
                        all_hairstyle_list.append(sub_key)

            for hairstyle in all_hairstyle_list:
                hairstyle_results = vectorstore.similarity_search_with_relevance_scores(query=hairstyle_keywords,k=1000,fetch_k=1000,filter={'gender':gender,'details':hairstyle})
                try:
                    avg_score = sum(score for _, score in hairstyle_results) / len(hairstyle_results)
                    face_score = 1 if hairstyle in faceshape_hairstyle_list else 0 
                    if season is not None:
                        seasonal_hairstyle_list = hairstyle_data['계절'][gender+season]
                        season_score = 1 if hairstyle in seasonal_hairstyle_list else 0 
                    else:
                        season_score = 0
                    results['hair'][hairstyle] = [avg_score,face_score,season_score] 
                    hair_max_score = max(avg_score,hair_max_score)
                    hair_min_score = min(avg_score,hair_min_score)
                except:
                    continue
            # weight.append(get_weight(hair_max_score,hair_min_score))
            weight['hair'] = get_weight(hair_max_score,hair_min_score)

        # 키워드 없으면 키워드 gender와 얼굴형으로 해서 RAG에서 doc 서치해서 반환
        else:
            hairstyle_keywords = f'{gender} {face_shape}'
            if hairlength_keywords:
                for hairstyle in faceshape_hairstyle_list:
                    for key, val in all_hairstyle_length.items():
                        for sub_key, sub_val in val.items():
                            if sub_key == hairstyle:
                                if hairlength_keywords in sub_val:
                                    hair_result = vectorstore.similarity_search_with_relevance_scores(query=hairstyle_keywords, k=1, fetch_k=1000, filter={'details':hairstyle,'gender':gender})
                                    result_docs[hairstyle] = [doc.page_content for doc,_ in hair_result]
            else:
                for hairstyle in faceshape_hairstyle_list:
                    hair_result = vectorstore.similarity_search_with_relevance_scores(query=hairstyle_keywords, k=1, fetch_k=1000, filter={'details':hairstyle,'gender':gender})
                    result_docs[hairstyle] = [doc.page_content for doc,_ in hair_result]

    if personal_color is not None :
        haircolor_list = hairstyle_data['퍼스널컬러'][personal_color]

        if haircolor_keywords is not None:    # 느낌에 관련된 키워드가 있으면 
            all_haircolor_list = hairstyle_data['전체 헤어스타일']['컬러']
            for haircolor in all_haircolor_list:
                haircolor_results = vectorstore.similarity_search_with_relevance_scores(query=haircolor_keywords,k=1000,fetch_k=1000,filter={"details":haircolor})
                try:
                    color_avg_score = sum(score for _, score in haircolor_results) / len(haircolor_results)
                    pc_score = 1 if haircolor in haircolor_list else 0
                    results['color'][haircolor] = [color_avg_score,pc_score]
                    color_max_score = max(color_avg_score,color_max_score)
                    color_min_score = min(color_avg_score,color_min_score)
                except:
                    continue
            # weight.append(get_weight(color_max_score,color_min_score))
            weight['color'] = get_weight(color_max_score,color_min_score)
        else:
            haircolor_keywords = personal_color
            for haircolor in haircolor_list:
                color_result = vectorstore.similarity_search_with_relevance_scores(query=haircolor_keywords, k=2, fetch_k=1000, filter={'details':haircolor})
                result_docs[haircolor] = [doc.page_content for doc,_ in color_result]
    

    # 가중치로 계산하기
    for idx, (category, value_dict) in enumerate(results.items()):
        for hairstyle, score_list in value_dict.items():
            if category == "color":
                scores[category].append([hairstyle, score_list[0] + weight[category] * score_list[1] ])
            else:
                scores[category].append([hairstyle, score_list[0] + weight[category] * score_list[1] + weight[category] * score_list[2]])
    
    if len(results['hair'])!=0 or len(results['color'])!=0:
        hairstyles = sorted(scores['hair'],key=lambda x:x[1], reverse=True)[:3]
        haircolors = sorted(scores['color'],key=lambda x:x[1], reverse=True)[:3]
        if len(hairstyles)!=0:
            hair_query = hairstyle_keywords if hairstyle_keywords is not None else f'{gender} {face_shape}' if (gender and face_shape) else '헤어스타일'
            for hairstyle, _ in hairstyles:
                hair_result = vectorstore.similarity_search_with_relevance_scores(query=hair_query, k=1, fetch_k=1000, filter={'details':hairstyle,'gender':gender})
                result_docs[hairstyle] = [doc.page_content for doc,_ in hair_result]
        if len(haircolors)!=0:
            color_query = haircolor_keywords if haircolor_keywords is not None else personal_color if personal_color else '헤어컬러'
            for haircolor, _ in haircolors:
                color_result = vectorstore.similarity_search_with_relevance_scores(query=color_query, k=1, fetch_k=1000, filter={'details':haircolor})
                result_docs[haircolor] = [doc.page_content for doc,_ in color_result]

    save_path = "rag_result_docs.txt"
    with open(save_path, "w", encoding="utf-8") as f:
        for key, docs in result_docs.items():
            f.write(f"### {key}\n")
            for idx, content in enumerate(docs, 1):
                f.write(f"[{idx}] {content}\n")
            f.write("\n")  # 구분 줄

    print(f"[INFO] result_docs saved to: {save_path}")

    summary = "사용자 질문에 초점을 맞춰서 반환된 문서를 참고해서 질문의 의도와 직접적으로 관련된 답변만 해봐"

    return result_docs, summary

################################cache####

#qa_cache를 인자로 보내서 캐시에서 답변 검색

def hairstyle_recommendation(model, image_base64, faceshape_keywords=None, gender_keywords=None, personalcolor_keywords=None, season=None, hairstyle_keywords=None, haircolor_keywords=None, hairlength_keywords=None, status_callback=None, qa_cache=None):
    if status_callback:
        status_callback("추천 헤어스타일 검색 중...")

    if image_base64.startswith('data:image'):
        image_data = base64.b64decode(image_base64.split(',')[1])
    else:
        image_data = base64.b64decode(image_base64)

    img = Image.open(BytesIO(image_data))
    img_array = np.array(img)
    balanced_array = grayworld_white_balance(img_array)
    img = Image.fromarray(balanced_array)

    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
        img.save(temp_file.name)
        temp_path = temp_file.name

    try:
        result = stone.process(temp_path, image_type='color',return_report_image=False,tone_palette='perla')
        skin_tone = skin_tone_choice(result)
        if personalcolor_keywords is not None:
            personal_color = personalcolor_keywords
        else:
            personal_color = classify_personal_color(skin_tone)
        # 만약 gender_keywords, faceshape_keywords가 None이 아닐 경우
        if gender_keywords is None and faceshape_keywords is None:
            face_shape, gender = get_face_shape_and_gender(model, temp_path)
        elif gender_keywords is not None and faceshape_keywords is None:
            face_shape, _ = get_face_shape_and_gender(model,temp_path)
            gender = gender_keywords
        elif gender_keywords is None and faceshape_keywords is not None:
            _, gender = get_face_shape_and_gender(model,temp_path)
            face_shape = faceshape_keywords
        else:
            face_shape = faceshape_keywords
            gender = gender_keywords
            
        # 캐시에서 답변 검색
        order = ['gender_keywords', 'faceshape_keywords', 'hairlength_keywords', 'season', 'hairstyle_keywords', 'haircolor_keywords', 'personalcolor_keywords']
        kwargs = {'gender_keywords': gender_keywords, 'faceshape_keywords': faceshape_keywords, 'hairlength_keywords': hairlength_keywords, 'season': season, 'hairstyle_keywords': hairstyle_keywords, 'haircolor_keywords': haircolor_keywords, 'personalcolor_keywords': personalcolor_keywords}
        question_parts = []
        for key in order:
            if key in kwargs and kwargs[key] is not None:
                    question_parts.append((key, str(kwargs[key])))
                    
        question_list = []
        meta_list = []
        for question in question_parts:
            if question[0] == 'hairstyle_keywords' or question[0] == 'haircolor_keywords':
                question_list.append(question[1])
            else:
                meta_list.append(question[1])
        cache_question = " ".join(question_list) #(가벼운, 붉은)
        cache_meta = " ".join(meta_list) #(female, round, ...)

        if question_parts and cache_question:
            # 캐시에서 답변 검색
            
            cached_doc = qa_cache.get_answer(cache_question, cache_meta)
            if cached_doc:
                # 문자열 형태의 최종 답변을 바로 반환
                cached_answer = cached_doc.metadata['answer']
                print("cached_answer", cached_answer)
                return (True,cached_answer,cache_question,cache_meta)
            
                
        with open("config/hairstyle_list.json", "r", encoding="utf-8") as f:
            hairstyle_data = json.load(f)

        with open("config/hairstyle_length.json", "r", encoding="utf-8") as f:
            hairstyle_length = json.load(f)

        scores = {'hair':[],'color':[]}
        results = {'hair':{},'color':{}}
        all_hairstyle_list = []
        all_hairstyle_length = hairstyle_length['헤어스타일'][gender]

        if hairlength_keywords:
            for key, val in all_hairstyle_length.items():
                for sub_key, sub_val in val.items():
                    if hairlength_keywords in sub_val:
                        all_hairstyle_list.append(sub_key)
        else:
            for key, val in all_hairstyle_length.items():
                for sub_key, sub_val in val.items():
                    all_hairstyle_list.append(sub_key)

        all_haircolor_list = hairstyle_data['전체 헤어스타일']['컬러']
        faceshape_hairstyle_list = hairstyle_data['얼굴형'][gender+face_shape]
        haircolor_list = hairstyle_data['퍼스널컬러'][personal_color]
        hair_max_score, hair_min_score = -inf, inf
        color_max_score, color_min_score = -inf, inf     
        
        if season is not None:
            seasonal_hairstyle_list = hairstyle_data['계절'][gender+season]
        
        if hairstyle_keywords is None:
            hairstyle_keywords = f"{face_shape}, {'여자' if gender == 'Female' else '남자'}"

        for hairstyle in all_hairstyle_list:
            hairstyle_results = vectorstore.similarity_search_with_relevance_scores(query=hairstyle_keywords,k=1000,fetch_k=1000,filter={'gender':gender,'details':hairstyle})
            try:
                avg_score = sum(score for _, score in hairstyle_results) / len(hairstyle_results)
                face_score = 1 if hairstyle in faceshape_hairstyle_list else 0 
                if season is not None:
                    season_score = 1 if hairstyle in seasonal_hairstyle_list else 0 
                else:
                    season_score = 0
                results['hair'][hairstyle] = [avg_score,face_score,season_score] 
                hair_max_score = max(avg_score,hair_max_score)
                hair_min_score = min(avg_score,hair_min_score)
            except:
                continue

        if haircolor_keywords is None:
            haircolor_keywords = f"{personal_color}"

        for haircolor in all_haircolor_list:
            haircolor_results = vectorstore.similarity_search_with_relevance_scores(query=haircolor_keywords,k=1000,fetch_k=1000,filter={"details":haircolor})
            try:
                color_avg_score = sum(score for _, score in haircolor_results) / len(haircolor_results)
                pc_score = 1 if haircolor in haircolor_list else 0
                results['color'][haircolor] = [color_avg_score,pc_score]
                color_max_score = max(avg_score,color_max_score)
                color_min_score = min(avg_score,color_min_score)
            except:
                continue
        # 가중치 가져오기
        weight = [get_weight(hair_max_score,hair_min_score),get_weight(color_max_score, color_min_score)]
        
        # 가중치로 계산하기
        for idx, (category, value) in enumerate(results.items()):
            for hairstyle, score_list in value.items():
                if len(score_list) == 2:
                    scores[category].append([hairstyle, score_list[0] + weight[idx] * score_list[1] ])
                    continue
                scores[category].append([hairstyle, score_list[0] + weight[idx] * score_list[1] + weight[idx] * score_list[2]])

        hairstyles = sorted(scores['hair'],key=lambda x:x[1], reverse=True)[:3]
        haircolors = sorted(scores['color'],key=lambda x:x[1], reverse=True)[:3]

        hairstyle_docs = {}
        for hairstyle, _ in hairstyles:
            hair_result = vectorstore.similarity_search_with_relevance_scores(query=hairstyle_keywords, k=1, fetch_k=1000, filter={'details':hairstyle,'gender':gender})
            hairstyle_docs[hairstyle] = [doc.page_content for doc,_ in hair_result]

        haircolor_docs = {}
        for haircolor, _ in haircolors:
            color_result = vectorstore.similarity_search_with_relevance_scores(query=haircolor_keywords, k=1, fetch_k=1000, filter={'details':haircolor})
            haircolor_docs[haircolor] = [doc.page_content for doc,_ in color_result]

        faceshape_docs = {}
        korean_faceshape = get_faceshape(face_shape)
        faceshape_result = vectorstore.similarity_search_with_score(query=hairstyle_keywords,k=2,fetch_k=1000,filter={'details':korean_faceshape,'gender':gender})
        faceshape_docs[korean_faceshape] = [doc.page_content for doc,_ in faceshape_result]
        summary = f"이 사람의 얼굴형은 {face_shape}, 성별은 {gender}이고 퍼스널컬러는 {personal_color}입니다."

        output_dir = "recommendation_logs"
        os.makedirs(output_dir, exist_ok=True)

        def save_docs_to_txt(filename, docs_dict):
            path = os.path.join(output_dir, filename)
            with open(path, "w", encoding="utf-8") as f:
                for key, docs in docs_dict.items():
                    f.write(f"=== {key} ===\n")
                    for doc in docs:
                        f.write(doc)
                        f.write("\n\n")
            return path

        faceshape_path = save_docs_to_txt("faceshape_docs.txt", faceshape_docs)
        hairstyle_path = save_docs_to_txt("hairstyle_docs.txt", hairstyle_docs)
        haircolor_path = save_docs_to_txt("haircolor_docs.txt", haircolor_docs)

        print("Documents saved:")
        print(f"- {faceshape_path}")
        print(f"- {hairstyle_path}")
        print(f"- {haircolor_path}")

        return False,gender,face_shape,personal_color,summary, faceshape_docs, personal_color, hairstyle_docs, haircolor_docs
        
    finally:
        os.unlink(temp_path)


def hairstyle_generation(image_base64, hairstyle=None, haircolor=None, hairlength=None, client=None, status_callback=None):
    if status_callback:
        status_callback("이미지 생성 중...")

    if image_base64.startswith('data:image'):
        image_data = base64.b64decode(image_base64.split(',')[1])
    else:
        image_data = base64.b64decode(image_base64)

    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
        temp_file.write(image_data)
        temp_path = temp_file.name

    try:
        face = face_crop(image_file=temp_path)
        if face is None:
            return "ERROR <이미지에 다수의 얼굴이 감지되었습니다. 툴 호출 결과를 반환하지 않습니다. 사용자에게 안내해주세요.> ERROR"
        face_upscale = get_high_resolution(face)

        face_upscale_img = Image.fromarray(face_upscale)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_upscale:
            face_upscale_img.save(temp_upscale.name)
            processed_path = temp_upscale.name

        with open('config/reference.json', 'r', encoding='utf-8') as f:
            reference = json.load(f)

        with open('config/hairstyle_length.json', 'r', encoding='utf-8') as f:
            length_config = json.load(f)

        image = None
        hairstyle_img_path = None
        haircolor_path = None
        result_text = ""
        hairstyle_dict = reference.get("헤어스타일", {})


        if hairstyle:
            base_path = None
            gender_key = None
            category_key = None

            for gender in hairstyle_dict.keys():
                for category in hairstyle_dict[gender].keys():
                    if hairstyle in hairstyle_dict[gender][category]:
                        base_path = hairstyle_dict[gender][category][hairstyle]
                        gender_key = gender
                        category_key = category
                        break
                if base_path:
                    break

            if base_path:
                # hairstyle_length.json에서 지원 기장 리스트 가져오기
                gender_en = "Male" if gender_key == "남자" else "Female"
                supported_lengths = None

                if gender_en in length_config.get("헤어스타일", {}):
                    if category_key in length_config["헤어스타일"][gender_en]:
                        supported_lengths = length_config["헤어스타일"][gender_en][category_key].get(hairstyle, [])

                if supported_lengths:
                    # 기장 호환성 확인
                    if hairlength is None or hairlength in supported_lengths:
                        # 대표 기장이거나 지원하는 기장인 경우
                        if hairlength is None:
                            # 지원 기장 개수에 따라 선택
                            length_count = len(supported_lengths)
                            if length_count == 1:
                                selected_length = supported_lengths[0]
                            elif length_count == 2:
                                selected_length = supported_lengths[1]  # 마지막
                            elif length_count == 3:
                                selected_length = supported_lengths[1]  # 가운데 두 번째
                            else:  # 4개 이상
                                selected_length = supported_lengths[2]  # 세 번째
                            hairstyle_img_path = f"{base_path}/{selected_length}/{hairstyle}.jpg"

                            result_text = ""
                        else:
                            hairstyle_img_path = f"{base_path}/{hairlength}/{hairstyle}.jpg"
                            result_text = f"요청하신 기장은 {hairlength} 기장에 해당합니다. {hairstyle}의 {hairlength} 기장으로 합성한 이미지가 생성되었습니다."
                    else:
                        # 가장 가까운 기장 찾기
                        current_list, closest_length = search_close_length_category_from_list(supported_lengths, hairlength)
                        hairstyle_img_path = f"{base_path}/{closest_length}/{hairstyle}.jpg"
                        result_text = f"현재 {hairstyle}이 지원하는 기장은 {current_list} 입니다. 사용자의 요청과 가까운 {closest_length} 기장의 {hairstyle}을 생성했습니다."
                else:
                    # 지원 기장 정보가 없는 경우 기본 경로 사용
                    hairstyle_img_path = f"{base_path}/{hairstyle}.jpg"

        if haircolor:
            color_dict = reference.get("컬러", {})
            haircolor_path = color_dict.get(haircolor, None)

        if hairstyle_img_path and haircolor_path:
            prompt = """첫번째 이미지의 사람 헤어스타일을 두번째 이미지의 사람 헤어스타일로 바꾸고 세번째 이미지의 사람 헤어컬러를 적용해줘.
                        이미지를 생성할때 첫번째 이미지의 사람 그대로 생성하되 헤어스타일과 헤어컬러만 바뀌어야 해."""
            image = generate_image(client, prompt, image_path=processed_path, shape_path=hairstyle_img_path, color_path=haircolor_path)
        elif hairstyle_img_path and haircolor_path is None:
            prompt = """첫번째 이미지의 사람 헤어스타일을 두번째 이미지의 사람 헤어스타일로 적용해주고 헤어컬러는 기존 그대로 유지해줘.
                        이미지를 생성할때 첫번째 이미지의 사람 그대로 생성하되 헤어스타일만 바뀌어야 해."""
            image = generate_image(client, prompt, image_path=processed_path, shape_path=hairstyle_img_path)
        elif haircolor_path and hairstyle_img_path is None:
            prompt = """첫번째 이미지의 사람 헤어컬러만 두번째 이미지의 사람 컬러로 바꿔줘.
                        이미지를 생성할때 첫번째 이미지의 사람 그대로 생성하되 헤어컬러만 바뀌어야 해."""
            image = generate_image(client, prompt, image_path=processed_path, color_path=haircolor_path)

        if image:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_gen:
                temp_gen.write(image)
                temp_gen_path = temp_gen.name

            swapped_face = face_swap(processed_path, temp_gen_path)

            folder_path = "./results"
            path = len([file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))])
            with open(f"results/{path}.jpg", "wb") as f:
                f.write(swapped_face)

            get_3d(image_file=f"{path}.jpg", input_dir=folder_path)

        return (result_text if result_text else "이미지 생성 완료. 이제 답변을 생성하세요", image)

    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        if 'processed_path' in locals() and os.path.exists(processed_path):
            os.unlink(processed_path)
        if 'temp_gen_path' in locals() and os.path.exists(temp_gen_path):
            os.unlink(temp_gen_path)


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
        prompt=prompt,
        size="1024x1024"
    )

    image_base64 = result.data[0].b64_json
    image_bytes = base64.b64decode(image_base64)

    return image_bytes

# def web_search(query: str)->str:
#     TAVILY_API_KEY=os.getenv("TAVILY_API_KEY")
#     tool = TavilySearch(
#         max_results=10,
#         topic = 'general',
#         tavily_api_key=TAVILY_API_KEY,
#         include_answer=True,
#         search_depth='basic',
#     )
#     results = tool.invoke(query)
#     final_result = results['answer']
#     for content in results['results']:
#         final_result += f" {content['content']}"
#     return final_result

def search_compatible_length(length, hairstyle_path):

    if length in hairstyle_path.keys():
        return True
    else:
        return False

def search_close_length_category(hairstyle_path, length):

    length_dict = {'숏': 0, '단발': 1, '중단발': 2, '미디엄': 3, '장발': 4}
    idx = length_dict.get(length)
    current_length_list = [(k, length_dict.get(k)) for k in hairstyle_path.keys()]  # [숏, 중단발]
    closest_length, _ = min(current_length_list, key=lambda x: abs(x[1] - idx))

    return f",".join(hairstyle_path.keys()), closest_length

def search_close_length_category_from_list(supported_lengths, length):
    """
    지원하는 기장 리스트에서 요청한 기장과 가장 가까운 기장을 찾는 함수
    """
    length_dict = {'숏': 0, '단발': 1, '중단발': 2, '미디엄': 3, '장발': 4}
    requested_idx = length_dict.get(length, 0)

    # 지원하는 기장들을 (기장명, 인덱스) 튜플 리스트로 변환
    length_list = [(l, length_dict.get(l, 0)) for l in supported_lengths]

    # 요청한 기장과의 차이가 가장 작은 기장 찾기
    closest_length, _ = min(length_list, key=lambda x: abs(x[1] - requested_idx))

    return ", ".join(supported_lengths), closest_length
 

def get_tool_list(*args):
    tools = load_tools(['dalle-image-generator'])
    tools.extend(args)
    return tools