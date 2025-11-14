from langchain_core.tools import tool
from langchain_core.prompts import  PromptTemplate
# from langchain.agents import create_openai_tools_agent
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_classic.agents import AgentExecutor, create_react_agent
from model.model_load import load_openai
from model.tools import hairstyle_recommendation, web_search, get_tool_list
import os
from model.model_load import use_endpoint

# ReAct 프롬프트 템플릿
react_prompt = PromptTemplate.from_template(
    """You are a hairstyle recommendation assistant.

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

When user provides an image and asks for hairstyle recommendations:
- Use hairstyle_recommendation_tool
- Then recommend 3 suitable hairstyles

When user asks to change/modify hairstyle:
- Use Dall-E-Image-Generator tool ONLY ONCE
- Immediately provide Final Answer after getting the image URL

**Important**:
you have to answer in Korean.

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
)

class HairstyleAgent:
    """헤어스타일 추천 Agent - 각 인스턴스가 독립적인 이미지 저장소를 가짐"""
    
    def __init__(self, model):
        """
        Args:
            model: IdentiFace 모델 (얼굴 분석용)
        """
        self.model = model
        self.current_image_base64 = None  # 인스턴스별 이미지 저장
        self.dalle_called = False  # DALL-E 호출 추적
        self.agent = self._build_agent()
    
    def _build_agent(self):
        """내부 agent 생성"""
        # Qwen VL 사용
        llm = use_endpoint(model_name="Qwen/Qwen3-VL-30B-A3B-Instruct", token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))
        
        # Tool 정의 - self.current_image_base64 사용
        @tool
        def hairstyle_recommendation_tool(action: str = "analyze"):
            """
            Analyzes the user's face from the provided image.
            Returns personal color, face shape, and gender information.
            Call this when user provides an image asking for hairstyle recommendations.
            """
            if self.current_image_base64 is None:
                return "오류: 이미지가 제공되지 않았습니다."
            print(f"[INFO] Tool 실행: Base64 길이 = {len(self.current_image_base64)}")
            return hairstyle_recommendation(self.model, self.current_image_base64)
        
        @tool
        def web_search_tool(query: str) -> str:
            """웹 검색 도구"""
            return web_search(query)
        
        
        tools = get_tool_list(hairstyle_recommendation_tool, web_search_tool)

        # ReAct Agent 생성 (Qwen 호환)
        agent = create_react_agent(llm, tools, react_prompt)

        # AgentExecutor 생성
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=5,
            max_execution_time=60,
            handle_parsing_errors=True,
        )

        # 세션 기록
        store = {}
        def get_session_history(session_ids):
            if session_ids not in store:
                store[session_ids] = ChatMessageHistory()
            return store[session_ids]

        agent_with_chat_history = RunnableWithMessageHistory(
            agent_executor,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

        return agent_with_chat_history
    
    def invoke(self, inputs, config=None, **kwargs):
        """
        Agent 실행 - 입력에서 이미지를 자동으로 추출
        
        Args:
            inputs: {"input": [HumanMessage(...)]} 형식
            config: {"configurable": {"session_id": "..."}} 형식
        """
        # 입력에서 이미지 추출
        if 'input' in inputs:
            messages = inputs['input']
            for msg in messages:
                if hasattr(msg, 'content') and isinstance(msg.content, list):
                    for content in msg.content:
                        if isinstance(content, dict) and content.get('type') == 'image_url':
                            self.current_image_base64 = content['image_url']['url']
                            print(f"[INFO] 이미지 감지! Base64 길이: {len(self.current_image_base64)}")
                            break
        
        # 원래 agent 실행
        return self.agent.invoke(inputs, config, **kwargs)


def build_agent(model):
    """
    HairstyleAgent 인스턴스를 생성하여 반환
    
    Args:
        model: IdentiFace 모델
        
    Returns:
        HairstyleAgent 인스턴스
    """
    return HairstyleAgent(model)

