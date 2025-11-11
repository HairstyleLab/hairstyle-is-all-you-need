import os
from dotenv import load_dotenv
from langchain.tools import Tool
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_tavily import TavilySearch
import stone
from google.colab.patches import cv2_imshow 
from json import dumps 



def get_tool_list():
    TAVILY_API_KEY=os.getenv("TAVILY_API_KEY")

    tools = []
    
    search = TavilySearch(
        max_results=10,
        topic = 'general',
        tavily_api_key=TAVILY_API_KEY,
        include_answer=True,
        search_depth='basic',
    )

    tools.append(
        get_facial_data_tool()
    )
    
    tools.append(
        Tool(
            name='hairstyle_features_tool',
            func=search.invoke(),
            description='계절별 추천 헤어스타일과 헤어스타일의 특징을 제공하는 도구입니다. 계절이나 헤어스타일을 입력받아 해당 헤어스타일의 특징을 설명합니다.',
            return_direct=False
        )
    )

    return tools


class FacialDataTool(Tool):
    name = "facial_data_tool"
    description = "얼굴형과 성별, 피부톤을 인식하는 도구입니다. 이미지를 입력받아 얼굴형과 성별, 피부색깔을 분석합니다."
    inputs=['image']
    outputs=['text']

    def __call__(self, image_path):
        facial_data = facial_data_analysis(image_path)
        skintone = skin_tone_analysis(image_path)    
        result = f"분석 결과: 얼굴형 - {facial_data[0]}, 성별 - {facial_data[1]}, 피부톤 - {skintone}"
        return result

def skin_tone_analysis(image_path):
    result = stone.process(image_path, image_type="color", return_report_image=False,tone_palette='yadon-ostfeld')  
    skintone_data = dumps(result)
    return skintone_data['faces'][0]['dominant_colors'][0]['color']  

def facial_data_analysis(image_path):
    result = []

    return result

def get_facial_data_tool():
    return FacialDataTool()




############# Backup Tool Code #############
from langchain.agents import tool

@tool
def hairstyle_recommandation_tool(model, image_path: str) -> str:
    """이미지를 입력받아 성별, 얼굴형, 피부톤을 분석하고 어울리는 헤어스타일을 문서에서 검색하여 반환하는 도구입니다."""
    result = stone.process(image_path, image_type="color", return_report_image=False,tone_palette='yadon-ostfeld')
    skin_tone = result['faces'][0]['dominant_colors'][0]['color']
    shape, gender = get_face_shape_and_gender(model, image_path)
    return skin_tone, shape, gender

@tool
def facial_shape_and_gender_tool(image_path: str) -> str:
    """얼굴형과 성별을 인식하는 도구입니다. 이미지를 입력받아 얼굴형과 성별을 분석합니다."""
    result = []   
    return result # [ 얼굴형 , 성별 ] 형태로 반환

@tool
def skin_tone_tool(image_path: str) -> str:
    """피부톤을 인식하는 도구입니다. 이미지를 입력받아 피부색깔을 분석합니다."""
    result = stone.process(image_path, image_type="color", return_report_image=False,tone_palette='yadon-ostfeld')
    return result['faces'][0]['dominant_colors'][0]['color']  

@tool
def web_search_tool(query:str)->str:
    """계절별 추천 헤어스타일과 헤어스타일의 특징을 제공하는 도구입니다. 계절이나 헤어스타일을 입력받아 해당 헤어스타일의 특징을 설명합니다."""
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

def get_tool_list():
    tools = [facial_shape_and_gender_tool, skin_tone_tool, web_search_tool]
    return tools