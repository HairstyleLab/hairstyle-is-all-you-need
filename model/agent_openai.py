from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
# from langchain.agents import create_openai_tools_agent
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_classic.agents import AgentExecutor,create_openai_tools_agent
from model.model_load import load_openai
from model.tools import hairstyle_recommendation, web_search, get_tool_list

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a hairstyle recommendation assistant.

When user provides an image and asks for hairstyle recommendations:
- Call hairstyle_recommendation_tool() without any parameters
- Then recommend 3 suitable hairstyles based on the result

When user asks to change/modify hairstyle:
- Use DALL-E tool

For text-only questions:
- Recommend 3 trending hairstyles
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("placeholder", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
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
        self.agent = self._build_agent()
    
    def _build_agent(self):
        """내부 agent 생성"""
        llm = load_openai(model_name="gpt-4o", temperature=0)
        
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

        # Agent 생성
        agent = create_openai_tools_agent(llm, tools, prompt)

        # AgentExecutor 생성
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=10,
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

