import os
import stone
from langchain_classic.agents import load_tools
from langchain_tavily import TavilySearch
from model.utils import get_face_shape_and_gender


def skin_tone_choice(result):
    dominant_result = tuple(int(result['faces'][0]['dominant_colors'][0]['color'].lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    nondominant_result = tuple(int(result['faces'][0]['dominant_colors'][1]['color'].lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    d1,_,_ = dominant_result
    n1,_,_ = nondominant_result
    if d1 > n1:
        return dominant_result
    else:
        return nondominant_result

def hairstyle_recommendation(image_path:str)->tuple:
    result = stone.process(image_path, image_type="color", return_report_image=False,tone_palette='yadon-ostfeld')
    skin_tone = skin_tone_choice(result)
    shape, gender = get_face_shape_and_gender(image_path)
    print(skin_tone, shape, gender)
    return (skin_tone, shape, gender)

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
    tools.extend(list[args])
    return tools