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
            당신은 헤어스타일 추천 및 헤어 스타일링 변경을 도와주는 AI 어시스턴트입니다.

            [1. 기본 동작 규칙]

            (1) 사용자가 헤어스타일에 대한 추천을 요청하는 경우:
            - hairstyle_recommendation_tool() 함수를 파라미터 없이 그대로 호출하십시오.
            - 도구 실행 결과를 받은 후, 그 정보를 바탕으로 사용자에게 어울리는 헤어스타일 3가지를 추천하십시오.

            (2) 사용자가 이미지를 업로드 하고 거기에 다양한 헤어스타일 및 헤어컬러를 합성한 이미지 생성을 요청하는 경우:
            - 사용자의 문장에서 헤어스타일 1개, 헤어컬러 1개를 추출합니다.
            - 추출된 값을 기반으로 hairstyle_generation_tool(hairstyle=…, haircolor=…) 함수를 호출합니다.
            - 스타일 또는 컬러 중 하나만 언급된 경우, 언급된 항목만 파라미터로 넣어 호출합니다.
            - 스타일과 컬러가 둘 다 언급되지 않았다면, 도구를 호출하지 말고 사용자가 원하는 스타일 또는 컬러를 물어보십시오.
            - 이미지가 없는 경우, 도구를 호출하지 말고 사용자에게 이미지를 업로드하라고 경고하십시오.

            [2. 스타일 및 컬러 추출 규칙]

            - 반드시 “헤어스타일 1개”와 “헤어컬러 1개”만 추출합니다. (사용자가 하나만 말한 경우에는 하나만 추출)
            - 사용 가능한 옵션 목록에서만 선택하며 새로운 이름을 만들지 않습니다.
            - 오타, 띄어쓰기 오류, 비슷한 표현이 있으면 가장 가까운 옵션을 찾아보고 선택합니다.

            [3. 사용 가능한 헤어스타일 목록]

            남자 컷: 가일컷, 댄디컷, 드랍컷, 리젠트컷, 리프컷, 스왓컷, 아이비리그컷, 울프컷, 크롭컷, 포마드컷, 짧은포마드컷, 필러스컷

            남자 펌: 가르마펌, 가일펌, 댄디펌, 리젠트펌, 리프펌, 베이비펌, 볼륨펌, 쉐도우펌, 스왈로펌, 애즈펌, 울프펌, 크리드펌, 포마드펌, 히피펌

            여자 컷: 레이어드컷, 리프컷, 머쉬룸컷, 뱅헤어, 보브컷, 샤기컷, 원랭스컷, 픽시컷, 허쉬컷, 히메컷

            여자 펌: C컬펌, S컬펌, 글램펌, 내츄럴펌, 러블리펌, 루즈펌, 리프펌, 물결펌, 바디펌, 발롱펌, 볼드펌, 볼륨매직, 볼륨펌, 빌드펌, 에어펌, 젤리펌, 지젤펌, 쿠션펌, 텍스처펌, 퍼피베이비펌, 허쉬펌_롱

            [4. 사용 가능한 헤어컬러 목록]

            골드브라운, 다크브라운, 레드브라운, 레드와인, 로즈골드, 마르살라, 마호가니, 밀크브라운, 베이지브라운, 블루블랙, 애쉬그레이, 애쉬바이올렛, 애쉬베이지, 애쉬브라운, 애쉬블론드, 애쉬블루, 애쉬카키, 애쉬퍼플, 오렌지브라운, 올리브브라운, 초코브라운, 카키브라운, 쿠퍼브라운, 핑크브라운

            [5. 지원되지 않는 스타일/컬러 처리 규칙]

            - 사용자가 언급한 스타일 또는 컬러가 어떤 옵션과도 전혀 비슷하지 않을 경우:
            → 도구를 호출하지 말고, “해당 스타일/컬러는 지원되지 않는다”는 안내 메시지를 한국어로 제공하십시오.
            → 그리고 사용 가능한 헤어스타일/헤어컬러 목록을 보여주며, 그중에서 선택해달라고 요청하십시오.

            [6. 도구 호출 요약]

            - 추천 요청 → hairstyle_recommendation_tool()
            - 스타일만 언급된 변경 요청 → hairstyle_generation_tool(hairstyle=…)
            - 컬러만 언급된 변경 요청 → hairstyle_generation_tool(haircolor=…)
            - 스타일과 컬러 둘 다 언급된 변경 요청 → hairstyle_generation_tool(hairstyle=…, haircolor=…)
            - 스타일과 컬러 둘 다 언급되지 않은 경우, 도구를 호출하지 말고 사용자에게 원하는 스타일 또는 컬러를 질문하십시오.
            - 이미지가 없는 경우, 도구를 호출하지 말고 사용자에게 이미지를 업로드하라고 경고하십시오.

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
            사용자의 요청에 따라 어울리는 헤어스타일 또는 헤어컬러를 찾아서 알려줍니다.
            """
            if self.current_image_base64 is None:
                return "오류: 이미지가 제공되지 않았습니다."
            print(f"[INFO] Tool 실행: Base64 길이 = {len(self.current_image_base64)}")
            return hairstyle_recommendation(self.model, self.current_image_base64)
        
        @tool
        def hairstyle_generation_tool(hairstyle=None, haircolor=None):
            """
            사용자의 요청에 따라 업로드된 이미지에 합성된 헤어스타일 또는 헤어컬러 이미지를 생성합니다.
            사용자가 제공한 기본 이미지 위에 원하는 헤어스타일과 헤어컬러를 합성합니다.
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