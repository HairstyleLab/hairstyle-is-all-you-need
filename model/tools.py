import os
import stone
from langchain_classic.agents import tool
from langchain_classic.agents import load_tools
from langchain_tavily import TavilySearch
from model.utils import get_face_shape_and_gender

@tool
def hairstyle_recommendation_tool(model, image_path:str)->tuple:
    """이미지를 입력받아 성별, 얼굴형, 피부톤을 분석하고 어울리는 헤어스타일을 문서에서 검색하여 반환하는 도구입니다."""
    result = stone.process(image_path, image_type="color", return_report_image=False,tone_palette='yadon-ostfeld')
    skin_tone = result['faces'][0]['dominant_colors'][0]['color']
    shape, gender = get_face_shape_and_gender(model, image_path)
    print(skin_tone, shape, gender)
    return (skin_tone, shape, gender)

@tool
def web_search_tool(query:str)->str:
    """검색이 필요한 경우 웹 검색을 수행할 수 있는 도구입니다. 검색된 결과를 반환합니다."""
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
    tools = load_tools(['dalle-image-generator'])
    custom_tools =  [hairstyle_recommendation_tool, web_search_tool]
    tools.extend(custom_tools)
    return tools