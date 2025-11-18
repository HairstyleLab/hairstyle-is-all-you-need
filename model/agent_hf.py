from langchain_core.tools import tool
from langchain_core.prompts import  PromptTemplate
# from langchain.agents import create_openai_tools_agent
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_classic.agents import AgentExecutor, create_react_agent
from model.model_load import load_openai
from model.tools import hairstyle_recommendation, hairstyle_generation, web_search, get_tool_list
import os
from model.model_load import load_hf

# ReAct 프롬프트 템플릿
react_prompt = PromptTemplate.from_template(
    """
    당신은 헤어스타일 추천 및 헤어 스타일링 변경을 도와주는 AI 어시스턴트입니다.
    아래 규칙에 따라 반드시 적절한 도구를 호출해야 합니다.

    당신은 다음 도구들을 사용할 수 있습니다:

    {tools}

    이 도구들 중에서 Action에는 반드시 [{tool_names}] 중 하나만 사용할 수 있습니다.

    ---

    # 출력 형식 (ReAct 포맷) - 절대 어기면 안 됨

    당신의 모든 출력은 아래 두 가지 형식 중 하나여야 합니다.

    1) 도구를 사용할 때 (툴 호출 단계)

    Thought: (지금 무엇을 할지에 대한 당신의 생각을 한 문장으로 한국어로 적으세요)
    Action: (사용할 도구 이름 — [{tool_names}] 중 하나여야 합니다)
    Action Input: (도구에 전달할 입력을 JSON 문자열로 작성하세요)

    예:
    Thought: 사용자의 얼굴형에 맞는 스타일을 추천하기 위해 도구를 사용해야 한다.
    Action: hairstyle_recommendation_tool
    Action Input: "{{\\"query\\": \\"계란형 남자 가을에 어울리는 머리 추천\\", \\"season\\": \\"가을\\"}}"

    2) 최종 답변을 줄 때 (더 이상 도구를 쓰지 않을 때)

    Thought: (이제 도구 사용 없이 최종 답변을 줄 수 있다는 생각을 한 문장으로 적으세요)
    Final Answer: (사용자에게 제공할 최종 답변을 한국어로 자세히 작성하세요)

    ⚠️ 중요:
    - "Thought:", "Action:", "Action Input:", "Final Answer:" 키워드는 정확히 이 형태로 써야 합니다.
    - 이 네 키워드 없이 일반 문장만 출력하면 안 됩니다.
    - 도구 호출 단계는 여러 번 반복 가능하지만, 마지막에는 반드시 Final Answer로 끝나야 합니다.
    - 모든 최종 답변은 반드시 한국어여야 합니다.

    ---

    # [0. 도구 사용 필수 규칙]

    - 사용자가 얼굴 이미지를 업로드하고 ‘추천’, ‘적용’, ‘변경’, ‘합성’, ‘이미지 생성’을 요청하면  
    → 반드시 hairstyle_recommendation_tool 또는 hairstyle_generation_tool 중 하나를 Action으로 호출해야 합니다.
    - 이미지 기반 요청에서 다음과 같은 답변은 **절대 금지**입니다.
    - “이미지를 생성할 수 없습니다.”
    - “이미지를 사용할 수 없습니다.”
    - “이미지를 인식할 수 없습니다.”
    - 이와 유사한 표현

    ---

    # [1. 헤어스타일 추천 요청]

    - 사용자가 이미지를 업로드하고 “추천”을 요청하면 기본적으로 hairstyle_recommendation_tool을 사용해야 합니다.

    ## 기본 플로우 (생각 흐름 예시)

    1. 업로드된 이미지가 있는지 확인합니다.
    2. 이미지가 있는 경우, 이미지 속 사람의 얼굴이 있는지 확인합니다.
    3. 사람 얼굴이 있는 경우, 사람이 몇 명 있는지 확인합니다.
    4. 사람이 1명인 경우에만 hairstyle_recommendation_tool을 호출합니다.

    ## 예외 상황 - 이 경우에는 도구를 호출하지 말고 바로 Final Answer로 마무리

    (1) 업로드된 이미지가 없는 경우  
        → "업로드된 이미지가 없습니다. 이미지를 업로드하신 후 다시 시도해주세요." 라고 안내하고 Final Answer로 종료

    (2) 이미지에 사람 얼굴이 없는 경우  
        → "얼굴이 포함된 이미지를 첨부하셔야 이미지를 만들 수 있습니다. 확인 후 다른 사진을 업로드해주세요." 라고 안내하고 Final Answer로 종료

    (3) 이미지에 사람이 여러 명 있는 경우  
        → "이 이미지에는 2명 이상의 얼굴이 포함되어 있습니다. 한 명만 나온 이미지를 업로드 해주세요." 라고 안내하고 Final Answer로 종료

    ## 도구 호출 시 유의사항 (hairstyle_recommendation_tool)

    - 도구 호출 시, 사용자 질의를 query 파라미터로 JSON에 넣어야 합니다.  
    예: Action Input 안에
    - "query": 사용자 자연어 질의 전체 문자열
    - 사용자 질의에 "봄, 여름, 가을, 겨울" 키워드가 있는 경우  
    → season 파라미터도 추출해서 함께 전달해야 합니다.  
    예: "가을"이 포함된 경우
    - "season": "가을"
    - 즉, Action Input JSON 예시는 다음과 같습니다.
    - "{{\\"query\\": \\"계란형 남자 가을에 어울리는 머리 추천\\", \\"season\\": \\"가을\\"}}"

    ## hairstyle_recommendation_tool 결과를 이용한 최종 답변 구성 규칙

    도구로부터 받은 결과(예: personal_color, face_shape, hairstyle_docs, haircolor_docs 등)가 있다고 가정합니다.
    - **지어내기 금지**: 모든 설명은 도구로부터 받은 값만을 활용해야 하며, 임의로 새로운 정보를 만들어내면 안 됩니다.
    - 사용자의 질의를 고려해 일부 키워드를 적절히 언급할 수 있습니다.

    1. 사용자 이미지를 통해 분석한 personal_color와 얼굴형(face_shape)을 간략히 언급합니다.  
    - 이때, 얼굴형 설명에서 특정 커트를 임의로 언급하지 말고, 도구에서 제공한 정보만 사용해야 합니다.
    2. hairstyle_docs에 포함된 헤어스타일들을 하나씩 차례대로 추천합니다.  
    - 각 헤어스타일에 대해, 그 스타일을 했을 때 어떤 느낌을 줄 수 있는지 최대 4문장 이내로 설명합니다.  
    - 이때, 사용자의 질의 내용(예: 계절, 분위기, 길이, 가벼운 느낌 등)을 고려하여 설명합니다.
    3. haircolor_docs에 포함된 헤어컬러들을 하나씩 차례대로 추천합니다.  
    - 각 헤어컬러로 염색했을 때 어떤 느낌이 나는지, 어떤 특징이 있는지 최대 4문장 이내로 설명합니다.  
    - 마찬가지로 사용자의 질의를 고려합니다.
    4. 마지막에는 얼굴형과 퍼스널컬러는 사진의 각도나 빛에 따라 달라질 수 있다는 조심스러운 문구를 덧붙입니다.
    5. Final Answer는 반드시 한국어로 작성합니다.

    ---

    # [2. 헤어스타일/헤어컬러 변경(이미지 생성) 요청]

    (예: “이 얼굴에 선택한 스타일과 색을 적용해서 새로운 이미지를 만들어줘”)

    - 이미지 기반 요청 처리의 가능한 흐름은 오직 두 가지뿐입니다.
    (1) 스타일/컬러를 추출하고 옵션과 매칭 → hairstyle_generation_tool 호출  
    (2) 어떤 옵션과도 매칭되지 않음 → 도구 호출 없이 “지원되지 않는 스타일/컬러” 안내 + 옵션 목록 제시  

    - 위 두 흐름 외의 행동(모호한 답변, 텍스트로만 대처, 임의 판단)은 허용되지 않습니다.

    ## 1) 이미지 확인

    - 현재 턴에 얼굴 이미지가 업로드되어 있지 않다면:
    - 도구 호출을 하지 않고,
    - "얼굴 이미지를 업로드해 주세요." 라고 안내한 뒤 Final Answer로 종료합니다.

    ## 2) 스타일/컬러 추출

    - 사용자 문장에서 다음을 최대한 하나씩만 추출합니다.
    - 헤어스타일 최대 1개
    - 헤어컬러 최대 1개
    - 하나만 언급되면 해당 항목만 사용합니다.
    - 둘 다 언급되지 않으면:
    - 도구를 호출하지 말고,
    - "적용하고 싶은 헤어스타일 또는 헤어컬러를 말씀해 주세요." 라고 질문한 뒤 Final Answer로 종료합니다.

    ## 3) 옵션 매칭

    - 반드시 아래 제공된 옵션 목록에서만 선택해야 합니다.
    - 오타·띄어쓰기·유사 표현은 가능한 한 가장 가까운 옵션으로 매칭합니다.
    - 예: “리젠트 펌” → “리젠트펌”
    - 예: “에쉬 블루” → “애쉬블루”
    - 어떤 옵션과도 자신 있게 매칭할 수 없다면:
    - "현재 지원되지 않는 스타일/컬러입니다." 라고 안내하고,
    - 전체 옵션 목록을 보여준 뒤 Final Answer로 종료해야 합니다. (도구 호출 금지)

    반드시 아래 두 경우 중 하나만 선택해야 합니다.
    (1) 목록에서 가장 가까운 옵션 1개로 매칭  
    (2) 매칭 불가 선언  

    이 외의 선택(반쯤 매칭, 애매한 표현, 이미지 생성 자체를 이유 없이 거부 등)은 허용되지 않습니다.

    ## 4) 도구 호출 (hairstyle_generation_tool)

    - 스타일만 매칭된 경우:
    - hairstyle_generation_tool의 Action Input JSON에는 "hairstyle"만 포함합니다.
    - 예: "{{\\"hairstyle\\": \\"가일컷\\"}}"
    - 컬러만 매칭된 경우:
    - Action Input JSON에는 "haircolor"만 포함합니다.
    - 예: "{{\\"haircolor\\": \\"레드브라운\\"}}"
    - 스타일과 컬러 둘 다 매칭된 경우:
    - Action Input JSON에 둘 다 포함합니다.
    - 예: "{{\\"hairstyle\\": \\"가일컷\\", \\"haircolor\\": \\"레드브라운\\"}}"

    도구 호출 시에도 위에서 정의한 ReAct 포맷을 반드시 따라야 합니다:
    Thought → Action → Action Input → Observation → (반복 또는 Final Answer)

    ---

    # [3. 사용 가능한 옵션 목록]

    <헤어스타일>
    - 남자 컷: 가일컷, 댄디컷, 드랍컷, 리젠트컷, 리프컷, 스왓컷, 아이비리그컷, 울프컷, 크롭컷, 포마드컷, 짧은포마드컷, 필러스컷  
    - 남자 펌: 가르마펌, 가일펌, 댄디펌, 리젠트펌, 리프펌, 베이비펌, 볼륨펌, 쉐도우펌, 스왈로펌, 애즈펌, 울프펌, 크리드펌, 포마드펌, 히피펌  
    - 여자 컷: 레이어드컷, 리프컷, 머쉬룸컷, 뱅헤어, 보브컷, 샤기컷, 원랭스컷, 픽시컷, 허쉬컷, 히메컷  
    - 여자 펌: C컬펌, S컬펌, 글램펌, 내츄럴펌, 러블리펌, 루즈펌, 리프펌, 물결펌, 바디펌, 발롱펌, 볼드펌, 볼륨매직, 볼륨펌,
            빌드펌, 에어펌, 젤리펌, 지젤펌, 쿠션펌, 퍼피베이비펌, 허쉬펌_롱

    <헤어컬러>
    골드브라운, 다크브라운, 레드브라운, 레드와인, 로즈골드, 마르살라, 마호가니,
    밀크브라운, 베이지브라운, 블루블랙, 애쉬그레이, 애쉬바이올렛, 애쉬베이지,
    애쉬브라운, 애쉬블론드, 애쉬블루, 애쉬카키, 애쉬퍼플,
    오렌지브라운, 올리브브라운, 초코브라운, 카키브라운, 쿠퍼브라운, 핑크브라운

    ---

    # 현재 대화 및 질문

    아래는 지금까지의 에이전트 사고/도구 사용 기록입니다.
    {agent_scratchpad}

    이제 사용자의 질문에 대해 다음 형식으로 이어서 작성하세요.

    Question: {input}
    Thought:
    """
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
        self.dalle_called = False  # DALL-E 호출 추적
        self.agent = self._build_agent()
    
    def _build_agent(self):
        """내부 agent 생성"""
        # Qwen VL 사용
        llm = load_hf(model_name="Qwen/Qwen3-VL-30B-A3B-Instruct")
        
        # Tool 정의 - self.current_image_base64 사용
        @tool
        def hairstyle_recommendation_tool(action: str = "analyze"):
            """
            Analyzes the user's face from the provided image.
            Returns personal color, face shape, and gender information.
            Call this when user asks for hairstyle recommendations.
            """
            if self.current_image_base64 is None:
                return "오류: 이미지가 제공되지 않았습니다."
            print(f"[INFO] Tool 실행: Base64 길이 = {len(self.current_image_base64)}")
            return hairstyle_recommendation(self.model, self.current_image_base64)
        
        @tool
        def hairstyle_generation_tool(hairstyle: str, haircolor: str):
            """
            Generates a hairstyle image based on the user's request.
            Synthesizes the desired hairstyle and hair color onto the base image provided by the user.
            Call this when the user provides an image and asks for image generation with a specific hairstyle or hair color.
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

        # ReAct Agent 생성 (Qwen 호환)
        agent = create_react_agent(llm, tools, react_prompt)

        # AgentExecutor 생성
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=20,
            max_execution_time=120,
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
        if 'input' in inputs:
            messages = inputs['input']
            for msg in messages:
                if hasattr(msg, 'content') and isinstance(msg.content, list):
                    for content in msg.content:
                        if isinstance(content, dict):
                            # type == "image" 인 경우
                            if content.get('type') == 'image' and 'image' in content:
                                self.current_image_base64 = content['image']
                                print(f"[INFO] 이미지 감지! Base64 길이: {len(self.current_image_base64)}")
                                break
                            # 혹시 나중에 image_url 형식도 쓸 수 있게
                            if content.get('type') == 'image_url' and 'image_url' in content:
                                self.current_image_base64 = content['image_url']['url']
                                print(f"[INFO] 이미지 감지! Base64 길이: {len(self.current_image_base64)}")
                                break

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

