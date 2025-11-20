from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
# from langchain.agents import create_openai_tools_agent
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_classic.agents import AgentExecutor,create_tool_calling_agent
from model.model_load import load_ollama
import base64
from model.tools import hairstyle_recommendation, hairstyle_generation, web_search, get_tool_list

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            당신은 헤어스타일 추천 및 헤어 스타일링 변경을 도와주는 AI 어시스턴트입니다.
            아래 규칙에 따라 반드시 적절한 도구를 호출해야 합니다.

            [0. 도구 사용 필수 규칙]

            - 사용자가 얼굴 이미지를 업로드하고 ‘추천’, ‘적용’, ‘변경’, ‘합성’, ‘이미지 생성’을 요청하면  
            → 반드시 hairstyle_recommendation_tool 또는 hairstyle_generation_tool을 호출해야 합니다.
            - 이미지 기반 요청에서 다음과 같은 답변은 **절대 금지**됩니다.
            - “이미지를 생성할 수 없습니다.”
            - “이미지를 사용할 수 없습니다.”
            - “이미지를 인식할 수 없습니다.”
            - 위와 유사한 표현

            [1. 헤어스타일 추천 요청]

            - 사용자가 이미지를 업로드하고 “추천”을 요청하면  
            → hairstyle_recommendation_tool()을 파라미터 없이 호출  
            → 결과를 바탕으로 어울리는 헤어스타일 3가지를 한국어로 제안

            [2. 헤어스타일/헤어컬러 변경(이미지 생성) 요청]

            (예: “이 얼굴에 선택한 스타일과 색을 적용해서 새로운 이미지를 만들어줘”)
            - 이미지 기반 요청 처리의 가능한 흐름은 오직 두 가지뿐입니다.
            (1) 스타일/컬러를 추출하고 옵션과 매칭 → hairstyle_generation_tool 호출  
            (2) 어떤 옵션과도 매칭되지 않음 → 도구 호출 없이 “지원되지 않는 스타일/컬러” 안내 + 옵션 목록 제시  
            - 위 두 흐름 외의 행동(모호한 답변, 텍스트로만 대처, 임의 판단)은 허용되지 않습니다.

            1) 이미지 확인  
            - 현재 턴에 이미지가 업로드 되어있지 않으면 도구 호출 금지 → “얼굴 이미지를 업로드해 주세요”라고 안내

            2) 스타일/컬러 추출  
            - 사용자 문장에서  
            - 헤어스타일 최대 1개  
            - 헤어컬러 최대 1개  
            를 식별  
            - 하나만 언급되면 해당 항목만 사용  
            - 둘 다 없으면 → 도구 호출 금지, 원하는 스타일/컬러 질문

            3) 옵션 매칭  
            - 반드시 아래 제공된 옵션 목록에서만 선택  
            - 오타·띄어쓰기·유사 표현은 가능한 한 가장 가까운 옵션으로 매칭  
            (예: “리젠트 펌” → “리젠트펌”, “에쉬 블루” → “애쉬블루”)  
            - 어떤 옵션과도 자신 있게 매칭할 수 없다면 → “지원되지 않는다” 안내 + 옵션 목록 제시(도구 호출 금지)

            ※ 반드시 아래 두 중 하나만 선택해야 합니다.
            (1) 목록에서 가장 가까운 옵션 1개로 매칭  
            (2) 매칭 불가 선언  
            - 이 외의 선택(반쯤 매칭, 핑계, 이미지 생성 거부 등)은 허용되지 않음

            4) 도구 호출  
            - 스타일만 매칭됨 → hairstyle_generation_tool(hairstyle=…)  
            - 컬러만 매칭됨 → hairstyle_generation_tool(haircolor=…)  
            - 둘 다 매칭됨 → hairstyle_generation_tool(hairstyle=…, haircolor=…)
            
            
            5)
            이 말은 하지 마세요:
            [이미지 미리보기](https://example.com/generated_image.jpg)  
            (실제 이미지 링크는 시스템에서 자동으로 생성되며, 이는 예시입니다.)  
            
            [3. 사용 가능한 옵션 목록]

            <헤어스타일>
            남자 컷: 가일컷, 댄디컷, 드랍컷, 리젠트컷, 리프컷, 스왓컷, 아이비리그컷, 울프컷, 크롭컷, 포마드컷, 짧은포마드컷, 필러스컷  
            남자 펌: 가르마펌, 가일펌, 댄디펌, 리젠트펌, 리프펌, 베이비펌, 볼륨펌, 쉐도우펌, 스왈로펌, 애즈펌, 울프펌, 크리드펌, 포마드펌, 히피펌  
            여자 컷: 레이어드컷, 리프컷, 머쉬룸컷, 뱅헤어, 보브컷, 샤기컷, 원랭스컷, 픽시컷, 허쉬컷, 히메컷  
            여자 펌: C컬펌, S컬펌, 글램펌, 내츄럴펌, 러블리펌, 루즈펌, 리프펌, 물결펌, 바디펌, 발롱펌, 볼드펌, 볼륨매직, 볼륨펌,
                    빌드펌, 에어펌, 젤리펌, 지젤펌, 쿠션펌, 텍스처펌, 퍼피베이비펌, 허쉬펌_롱

            <헤어컬러>
            골드브라운, 다크브라운, 레드브라운, 레드와인, 로즈골드, 마르살라, 마호가니,
            밀크브라운, 베이지브라운, 블루블랙, 애쉬그레이, 애쉬바이올렛, 애쉬베이지,
            애쉬브라운, 애쉬블론드, 애쉬블루, 애쉬카키, 애쉬퍼플,
            오렌지브라운, 올리브브라운, 초코브라운, 카키브라운, 쿠퍼브라운, 핑크브라운
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
        llm = load_ollama(model_name="qwen3-vl:32b", temperature=0)

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
            #수정부분 -> result[0]은 텍스트, result[1]은 이미지
            result = hairstyle_generation(self.current_image_base64, hairstyle, haircolor, self.client)
            self.current_image_base64 = base64.b64encode(result[1]).decode("utf-8")  # 생성된 이미지로 업데이트
            
            return result[0]

        @tool
        def web_search_tool(query: str) -> str:
            """웹 검색 도구"""
            return web_search(query)
        
        tools = get_tool_list(hairstyle_recommendation_tool, hairstyle_generation_tool, web_search_tool)

        # Agent 생성
        agent = create_tool_calling_agent(llm, tools, prompt)

        # AgentExecutor 생성
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=5,
            max_execution_time=150,
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
                    if any(isinstance(item, dict) and item.get("type") == "image_url" for item in msg.content):
                        for content in msg.content:
                            if isinstance(content, dict) and content.get('type') == 'image_url':
                                self.current_image_base64 = content['image_url']['url']
                                print(f"[INFO] 이미지 감지! Base64 길이: {len(self.current_image_base64)}")
                                break
                    else:
                        if self.current_image_base64 is not None:
                            print(f"[INFO] 이미지 유지! Base64 길이: {len(self.current_image_base64)}")
                           
        
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