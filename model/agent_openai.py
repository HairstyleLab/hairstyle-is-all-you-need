from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
# from langchain.agents import create_openai_tools_agent
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_classic.agents import AgentExecutor,create_openai_tools_agent
from model.model_load import load_openai
from model.tools import hairstyle_recommendation, hairstyle_generation, web_search, get_tool_list

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a hairstyle recommendation assistant.

            When the user provides an image and asks for hairstyle recommendations:
            - Call hairstyle_recommendation_tool() without any parameters.
            - After receiving the tool result, recommend 3 suitable hairstyles based on the returned information.

            When the user asks to change or modify a hairstyle:
            - Call hairstyle_generation_tool with the extracted hairstyle and/or haircolor parameters.
            - If the user mentions a specific hairstyle or haircolor, extract those values accurately.

            === STYLE & COLOR EXTRACTION RULES ===
            You must extract exactly one hairstyle and one haircolor from the user's Korean question,
            using ONLY the predefined options below.

            <HAIRSTYLE_OPTIONS>
            남자 컷: 가일컷, 댄디컷, 드랍컷, 리젠트컷, 리프컷, 스왓컷, 아이비리그컷, 울프컷, 크롭컷, 포마드컷, 짧은포마드컷, 필러스컷
            남자 펌: 가르마펌, 가일펌, 댄디펌, 리젠트펌, 리프펌, 베이비펌, 볼륨펌, 쉐도우펌, 스왈로펌, 애즈펌, 울프펌, 크리드펌, 포마드펌, 히피펌
            여자 컷: 레이어드컷, 리프컷, 머쉬룸컷, 뱅헤어, 보브컷, 샤기컷, 원랭스컷, 픽시컷, 허쉬컷, 히메컷
            여자 펌: C컬펌, S컬펌, 글램펌, 내츄럴펌, 러블리펌, 루즈펌, 리프펌, 물결펌, 바디펌, 발롱펌, 볼드펌, 볼륨매직, 볼륨펌, 빌드펌, 에어펌, 젤리펌, 지젤펌, 쿠션펌, 텍스처펌, 퍼피베이비펌, 허쉬펌_롱
            </HAIRSTYLE_OPTIONS>

            <COLOR_OPTIONS>
            골드브라운, 다크브라운, 레드브라운, 레드와인, 로즈골드, 마르살라, 마호가니,
            밀크브라운, 베이지브라운, 블루블랙, 애쉬그레이, 애쉬바이올렛, 애쉬베이지,
            애쉬브라운, 애쉬블론드, 애쉬블루, 애쉬카키, 애쉬퍼플, 오렌지브라운,
            올리브브라운, 초코브라운, 카키브라운, 쿠퍼브라운, 핑크브라운
            </COLOR_OPTIONS>

            Rules:
            - Extract hairstyle and haircolor in the user's Korean input.
            - Select only from the options above; do not create new names or variations.
            - If the user's wording is a clear typo, spacing variation, or minor distortion of an existing option, map it to the closest valid option.
            - If the user uses a hairstyle or haircolor that cannot be reasonably matched to any predefined option, do NOT call hairstyle_generation_tool.
            Instead, politely explain in Korean that only registered hairstyles and colors can be used, show the available options,
            and ask the user to choose one hairstyle and/or one color from the lists.
            - If the user mentions only one of them, call hairstyle_generation_tool with only that parameter (do not include the other field).
            - If neither a hairstyle nor a haircolor is mentioned, ask the user to provide the desired hairstyle or haircolor before calling the tool.

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
    
    def __init__(self, model, client):
        """
        Args:
            model: IdentiFace 모델 (얼굴 분석용)
        """
        self.model = model
        self.client = client
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
        def hairstyle_generation_tool(hairstyle=None, haircolor=None):
            """
            Generates a hairstyle image based on the user's request.
            Synthesizes the desired hairstyle and hair color onto the base image provided by the user.
            Call this when the user provides an image and asks for a specific hairstyle or hair color.
            """
            if self.current_image_base64 is None:
                return "오류: 이미지가 제공되지 않았습니다."
            print(f"[INFO] Tool 실행: Base64 길이 = {len(self.current_image_base64)}")
            return hairstyle_generation(self.current_image_base64, hairstyle, haircolor, self.client)

        @tool
        def web_search_tool(query: str) -> str:
            """웹 검색 도구"""
            return web_search(query)
        
        tools = get_tool_list(hairstyle_recommendation_tool, hairstyle_generation_tool, web_search_tool)

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


def build_agent(model, client):
    """
    HairstyleAgent 인스턴스를 생성하여 반환
    
    Args:
        model: IdentiFace 모델
        
    Returns:
        HairstyleAgent 인스턴스
    """
    return HairstyleAgent(model, client)