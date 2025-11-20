from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
# from langchain.agents import create_openai_tools_agent
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_classic.agents import AgentExecutor,create_openai_tools_agent
from model.model_load import load_openai
from model.tools import hairstyle_recommendation, hairstyle_generation, web_search, get_tool_list,non_image_recommendation

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            당신은 헤어스타일 추천 및 헤어 스타일링 변경을 도와주는 AI 어시스턴트입니다.
            아래 규칙에 따라 반드시 적절한 도구를 호출해야 합니다.

            [0. 도구 사용 필수 규칙]

            - 사용자가 ‘추천’, ‘적용’, ‘변경’, ‘합성’, ‘이미지 생성’을 요청하면  
            → 반드시 hairstyle_recommendation_tool 또는 hairstyle_generation_tool을 호출해야 합니다.
            - 이미지 기반 요청에서 다음과 같은 답변은 **절대 금지**됩니다.
            - “이미지를 생성할 수 없습니다.”
            - “이미지를 사용할 수 없습니다.”
            - “이미지를 인식할 수 없습니다.”
            - 위와 유사한 표현

            [1. 이미지 없는 헤어스타일 추천 요청]
            - 사용자가 이미지 없이 헤어스타일 "추천"을 요청하면 non_image_recommendation_tool() 도구 호출

            **non_image_recommendation_tool() 도구 호출 시 기본 플로우**
           1. 사용자 질의에서 성별, 얼굴형, 퍼스널컬러, 하고 싶은 헤어스타일 키워드, 계절 키워드가 있는지 확인
           2. 사용자 질의에 성별과 얼굴형 언급이 있거나, 퍼스널컬러가 있으면 유의사항 확인해 파라미터로 전달하고 도구 호출
           3. 도구로부터 얻은 정보를 응답에 활용

           **도구 호출 시 유의사항**
           - 사용자 질의에 퍼스널 컬러에 대한 언급이 있는 경우, 퍼스널 컬러를 다음 리스트 중에서 찾아서 personal_color 파라미터로 전달 → non_image_recommendation_tool(personal_color=...)
             (퍼스널컬러 리스트: "봄 웜톤, 가을 웜톤, 겨울 쿨톤, 여름 쿨톤") 비슷한 말이 있으면 리스트 중에서 찾아서 전달
           - 사용자 질의에 얼굴형에 대한 언급이 있는 경우, 얼굴형 리스트를 참고해 영어로 바꾼 후 face_shape 파라미터로 전달 → non_image_recommendation_tool(face_shape=...)
             (얼굴형 리스트: "둥근형"→"Round", "사각형"→"Sqaure", "하트형"→"Heart", "계란형"→"Oval", "긴형"→"Oblong" )
           - 사용자 질의에 성별에 대한 언급이 있는 경우, 여자는 "Female" 남자는 "Male"를 gender 파라미터로 전달 → non_image_recommendation_tool(gender=...)
           - 도구 호출할 때 사용자 질의에 하고 싶은 머리 스타일에 관련된 키워드가 있는 경우, hairstyle_keywords 파라미터로 전달. 컬러는 제외. 키워드 없으면 전달 X→ non_image_recommendation_tool(hairstyle_keywords=...)
            (예) 질의: 여름이 되었으니까 가벼운 머리를 하고싶어. 층을 좀 냈으면 좋겠지만 너무 지저분하진 않으면 좋겠어 어두운 색이나 톤 다운된 색으로 염색도 하고 싶어 → hairstyle_keywords="가벼운, 층 내는, 지저분하지 않은"
           - 사용자 질의에 "봄, 여름, 가을, 겨울"의 키워드가 있는 경우, **계절**도 정확히 추출해 season 파라미터로 전달 → non_image_recommendation_tool(season=...)
           - 도구 호출할 때 사용자 질의에 하고 싶은 머리 컬러에 관련된 키워드가 있는 경우, haircolor_keywords 파라미터로 전달. 키워드 없으면 전달 X→ non_image_recommendation_tool(haircolor_keywords=...)
            (예) 질의: 여름이 되었으니까 가벼운 머리를 하고싶어. 층을 좀 냈으면 좋겠지만 너무 지저분하진 않으면 좋겠어 어두운 색이나 톤 다운된 색으로 염색도 하고 싶어 → haircolor_keywords="어두운, 톤 다운"

           **예외상황**
           1. 성별과 얼굴형, 퍼스널컬러가 모두 없는 경우, "성별과 얼굴형 또는 퍼스널컬러를 알려주셔야 헤어스타일 추천이 가능합니다. 다시 질의를 보내주세요"로 응답하고 마무리
           2. 성별과 얼굴형 둘 중 하나만 있는 경우, 나머지 하나를 알려주어야한다고 답변하고 마무리 
           (예)"성별을 알려주셔야 헤어스타일 추천이 가능합니다. 다시 질의를 보내주세요", "얼굴형을 알려주셔야 헤어스타일 추천이 가능합니다. 다시 질의를 보내주세요"

           **답변 순서**
           - 모든 설명은 지어내지말고 도구로부터 받은 값만을 활용해 생성. 사용자 질의를 일부 언급하며 자연스럽게 설명.
           1. 얼굴형과 헤어스타일이 있는 경우 얼굴형과 얼굴형 특징을 4문장 이내로 설명한 후, 어울리는 헤어스타일 3문장 이내로 설명. 
           - 얼굴형 설명에는 특정 커트를 언급하지 말고 지어내지말것. 도구로부터 받은 값만을 활용해 생성.
           2. result_docs에 있는 헤어스타일이나 헤어컬러를 하나씩 차례대로 추천.
           - 헤어스타일이 있으면, 각 헤어스타일을 하면 어떤 느낌을 줄 수 있는지 4문장 이내로 사용자 질의 내용을 고려해 자세히 설명. 
           - 헤어컬러가 있으면, 각 헤어컬러로 염색하면 어떤 느낌이 나는지 특징에 대해 4문장 이내로 사용자 질의 내용을 고려해 자세히 설명.
           4. 사진을 주시면 조금 더 세밀한 답변이 가능하다는 말로 마무리
           5. 모든 답변은 반드시 한국어여야함

           [2. 이미지 있는 헤어스타일 추천 요청]
            - 사용자가 이미지를 업로드하고 헤어스타일 “추천”을 요청하면 hairstyle_recommendation_tool() 도구 호출

            **기본 플로우**
            1. 업로드된 이미지가 있는지 확인
            2. 이미지가 있는 경우, 이미지 속 사람의 얼굴이 있는지 확인
            3. 사람 얼굴이 있는 경우, 사람이 몇 명 있나 확인
            3. 사람이 1명 있을 경우, hairstyle_recommendation_tool 도구 호출해 응답 생성에 활용

           **예외 상황**
           (1) 업로드된 이미지가 없는 경우, "업로드된 이미지가 없습니다. 이미지를 업로드하신 후 다시 시도해주세요."라고 응답 후 마무리
           (2) 이미지에 사람 얼굴이 없는 경우, "얼굴이 포함된 이미지를 첨부하셔야 이미지를 만들 수 있습니다. 확인 후 다른 사진을 업로드해주세요."라고 응답 후 마무리
           (3) 이미지에 사람이 여러명이 있는 경우, "이 이미지에는 2 명 이상의 얼굴이 포함되어 있습니다. 한 명만 나온 이미지를 업로드 해주세요."라고 응답 후 마무리

           **유의사항**
           - 도구 호출할 때 사용자 질의에 하고 싶은 머리에 관련된 키워드가 있는 경우, keywords 파라미터로 전달. 키워드 없으면 전달 X→ hairstyle_recommendation_tool(keywords=...)
            (예) 질의: 여름이 되었으니까 가벼운 머리를 하고싶어. 층을 좀 냈으면 좋겠지만 너무 지저분하진 않으면 좋겠어 어두운 색이나 톤 다운된 색으로 염색도 하고 싶어 → keywords="가벼운, 층 내는, 지저분하지 않은, 어두운, 톤 다운"
            
           - 사용자 질의에 "봄, 여름, 가을, 겨울"의 키워드가 있는 경우, **계절**도 정확히 추출해 season 파라미터로 전달 → hairstyle_recommendation_tool(keywords=..., season=...)

           **답변 순서**
           - 모든 설명은 지어내지말고 도구로부터 받은 값만을 활용해 생성. keywords가 있으면 이를 고려해 일부 언급하며 자연스럽게 설명.
           1. 사용자 이미지를 통해 분석한 personal_color와 얼굴형을 간략히 언급한 후, 얼굴형에 어울리는 헤어스타일 3문장 이내로 설명. 
           - 얼굴형 설명에는 특정 커트를 언급하지 말고 지어내지말것. 도구로부터 받은 값만을 활용해 생성.
           2. hairstyle_docs에 있는 헤어스타일을 하나씩 차례대로 추천. 
           - 각 헤어스타일을 하면 어떤 느낌을 줄 수 있는지 4문장 이내로 사용자 질의 내용을 고려해 자세히 설명. 
           3. haircolor_docs에 있는 헤어컬러를 하나씩 차례대로 추천.
           - 각 헤어컬러로 염색하면 어떤 느낌이 나는지 특징에 대해 4문장 이내로 사용자 질의 내용을 고려해 자세히 설명.
           4. 얼굴형과 퍼스널컬러는 사진의 각도나 빛에 따라 달라질 수 있다는 조심스러운 문구로 마무리
           5. 모든 답변은 반드시 한국어여야함


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
        self.last_inputs = None
        self.current_image_base64 = None  # 인스턴스별 이미지 저장
        self.agent = self._build_agent()
    
    def _build_agent(self):
        """내부 agent 생성"""
        llm = load_openai(model_name="gpt-4o", temperature=0)
        
        # Tool 정의 - self.current_image_base64 사용
        @tool
        def hairstyle_recommendation_tool(keywords:str, season=None, action: str = "analyze"):
            """
            사용자의 요청에 따라 어울리는 헤어스타일 또는 헤어컬러를 찾아서 알려줍니다.
            """
            if self.current_image_base64 is None:
                return "오류: 이미지가 제공되지 않았습니다."
            print(f"[INFO] Tool 실행: Base64 길이 = {len(self.current_image_base64)}")
            return hairstyle_recommendation(self.model, self.current_image_base64, keywords, season)

        @tool
        def non_image_recommendation_tool(face_shape=None, gender=None, personal_color=None, season=None, hairstyle_keywords=None, haircolor_keywords=None):
            """
            이미지 없이 사용자 요청에 따라 어울리는 헤어스타일 또는 헤어컬러를 찾아서 알려줍니다.
            """
            print(f"[INFO] TOOL 실행 -> 키워드 얼굴형:{face_shape}, 성별:{gender}, 퍼컬: {personal_color}, 계절: {season}, 키워드:{hairstyle_keywords} {haircolor_keywords}")
            return non_image_recommendation(face_shape, gender, personal_color, season, hairstyle_keywords, haircolor_keywords)
        
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
        
        tools = get_tool_list(hairstyle_recommendation_tool,non_image_recommendation_tool, hairstyle_generation_tool, web_search_tool)

        # Agent 생성
        agent = create_openai_tools_agent(llm, tools, prompt)

        # AgentExecutor 생성
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=20,
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
            self.last_inputs = messages
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