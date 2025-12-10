from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
# from langchain.agents import create_openai_tools_agent
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_classic.agents import AgentExecutor,create_openai_tools_agent
from model.model_load import load_openai
from model.tools import hairstyle_recommendation, hairstyle_generation, web_search, rag_search, get_tool_list, non_image_recommendation
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import OpenAIEmbeddings
from rag.qa_cache import QACache
import base64
from .system_prompt import sys_prompt
import ast


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            sys_prompt,
        ),
        ("placeholder", "{chat_history}"),
        ("placeholder", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

class HairstyleAgent:

    def __init__(self, model, client):

        self.model = model
        self.client = client
        self.last_inputs = None
        self.current_image_base64 = None  # 인스턴스별 이미지 저장
        self.gen_flag = False             # 이미지 생성했는지 여부
        self.status_callback = None

        # 캐시 관련 변수
        self.last_tool_params = None      # 마지막 Tool 호출 파라미터
        self.last_tool_cache_hit = False  # 캐시 히트 여부
        self.cached_final_answer = None   # 캐시된 최종 답변

        # QA 캐시 초기화
        self.qa_cache = self._init_qa_cache()
        self.agent = self._build_agent()
        
    def _init_qa_cache(self):
        try:
            embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
            qa_cache = QACache(
                json_path="rag/qa.json",
                embeddings=embeddings,
                vectorstore_path="rag/qa_vectorstore",
                similarity_threshold=0.85,
                batch_size=1
            )
            print("QA Cache 초기화 완료")
            return qa_cache
        except Exception as e:
            print(f"QA Cache 초기화 실패: {e}")
            return None
        
    def search_cache(self, **kwargs):
        """캐시에서 답변 검색 - 모든 키워드 인자를 받아서 처리"""
        if self.qa_cache:
            try:
                # 질문 문자열 생성 - 특정 순서로 정렬 (gender, face_shape, hairlength, season, hairstyle, haircolor, personal_color)
                order = ['gender', 'face_shape', 'hairlength_keywords', 'season', 'hairstyle_keywords', 'haircolor_keywords', 'personal_color']
                question_parts = []
                for key in order:
                    if key in kwargs and kwargs[key] is not None:
                        question_parts.append(str(kwargs[key]))

                if question_parts:
                    cache_question = " ".join(question_parts)

                    # 캐시에서 답변 검색
                    cached_doc = self.qa_cache.get_answer(cache_question)
                    if cached_doc:
                        print(f"[CACHE HIT] 캐시에서 답변 반환: {cache_question[:50]}...")
                        # 문자열 형태의 최종 답변을 바로 반환
                        cached_answer = cached_doc.metadata['answer']
                        return cached_answer
                    else:
                        print(f"[CACHE MISS] 캐시에 없음, 새로 추론: {cache_question[:50]}...")
                        return None

            except Exception as e:
                print(f"Cache 검색 실패: {e}")
        return None
                    
    def store_cache(self, answer, **kwargs):
        """캐시에 답변 저장 - 모든 키워드 인자를 받아서 처리"""
        if self.qa_cache:
            try:
                # 질문 문자열 생성 - 특정 순서로 정렬 (gender, face_shape, hairlength, season, hairstyle, haircolor, personal_color)
                order = ['gender', 'face_shape', 'hairlength_keywords', 'season', 'hairstyle_keywords', 'haircolor_keywords', 'personal_color']
                question_parts = []
                for key in order:
                    if key in kwargs and kwargs[key] is not None:
                        question_parts.append(str(kwargs[key]))

                if question_parts:
                    cache_question = " ".join(question_parts)

                    # 최종 답변(문자열) 저장
                    if isinstance(answer, str):
                        cache_answer = answer
                        self.qa_cache.add_qa(cache_question, cache_answer)
                        print("Cache Size:", self.qa_cache.get_cache_size())
                        print("Is Saved:", self.qa_cache.verify_saved(cache_question))
                        print(f"Cache 저장: {cache_question[:50]}...")
                    

            except Exception as e:
                print(f"Cache 저장 실패: {e}")
    
    def _build_agent(self):
        llm = load_openai(model_name="gpt-5-mini", temperature=0)
        
        # Tool 정의 - self.current_image_base64 사용
        @tool
        def hairstyle_recommendation_tool(season=None, hairstyle_keywords=None, haircolor_keywords=None, hairlength_keywords=None):
            """
            사용자의 요청에 따라 어울리는 헤어스타일 또는 헤어컬러를 찾아서 알려줍니다.
            """
            if self.current_image_base64 is None:
                return "오류: 이미지가 제공되지 않았습니다."
            print(f"[INFO] Tool 실행: Base64 길이 = {len(self.current_image_base64)}")

            # 캐시에서 최종 답변 검색
            cached = self.search_cache(season=season, hairstyle_keywords=hairstyle_keywords, haircolor_keywords=haircolor_keywords, hairlength_keywords=hairlength_keywords)

            if cached is not None:
                # 캐시된 답변을 그대로 반환 (이미 GPT가 생성한 최종 답변)
                self.last_tool_cache_hit = True
                self.cached_final_answer = cached
                return "캐시에서 답변을 찾았습니다. 이 정보를 바탕으로 답변해주세요: " + cached

            result = hairstyle_recommendation(self.model, self.current_image_base64, season, hairstyle_keywords, haircolor_keywords, hairlength_keywords, status_callback=self.status_callback)
            # 캐시 저장은 invoke()에서 최종 답변과 함께 수행
            self.last_tool_params = {'season': season, 'hairstyle_keywords': hairstyle_keywords, 'haircolor_keywords': haircolor_keywords, 'hairlength_keywords': hairlength_keywords}
            self.last_tool_cache_hit = False

            return result

        @tool
        def non_image_recommendation_tool(face_shape=None, gender=None, personal_color=None, season=None, hairstyle_keywords=None, haircolor_keywords=None, hairlength_keywords=None):
            """
            이미지 없이 사용자 요청에 따라 어울리는 헤어스타일 또는 헤어컬러를 찾아서 알려줍니다.
            """
            print(f"[INFO] TOOL 실행 -> 키워드 얼굴형:{face_shape}, 성별:{gender}, 퍼컬: {personal_color}, 계절: {season}, 키워드:{hairstyle_keywords} {haircolor_keywords}")

            # 캐시에서 최종 답변 검색
            cached = self.search_cache(season=season, hairstyle_keywords=hairstyle_keywords, haircolor_keywords=haircolor_keywords, hairlength_keywords=hairlength_keywords, gender=gender, face_shape=face_shape, personal_color=personal_color)

            if cached is not None:
                # 캐시된 답변을 그대로 반환 (이미 GPT가 생성한 최종 답변)
                self.last_tool_cache_hit = True
                self.cached_final_answer = cached
                return "캐시에서 답변을 찾았습니다. 이 정보를 바탕으로 답변해주세요: " + cached

            result = non_image_recommendation(face_shape, gender, personal_color, season, hairstyle_keywords, haircolor_keywords, hairlength_keywords, status_callback=self.status_callback)
            # 캐시 저장은 invoke()에서 최종 답변과 함께 수행
            self.last_tool_params = {'season': season, 'hairstyle_keywords': hairstyle_keywords, 'haircolor_keywords': haircolor_keywords, 'hairlength_keywords': hairlength_keywords, 'gender': gender, 'face_shape': face_shape, 'personal_color': personal_color}
            self.last_tool_cache_hit = False

            return result
          

        @tool
        def hairstyle_generation_tool(hairstyle=None, haircolor=None):
            """
            사용자의 요청에 따라 업로드된 이미지에 합성된 헤어스타일 또는 헤어컬러 이미지를 생성합니다.
            사용자가 제공한 기본 이미지 위에 원하는 헤어스타일과 헤어컬러를 합성합니다.
            """
            if self.current_image_base64 is None:
                return "오류: 이미지가 제공되지 않았습니다."
            print(f"[INFO] Tool 실행: Base64 길이 = {len(self.current_image_base64)}")

            if res := hairstyle_generation(self.current_image_base64, hairstyle, haircolor, self.client, status_callback=self.status_callback):
                self.gen_flag = True
            self.current_image_base64 = base64.b64encode(res[1]).decode('utf-8')

            return res[0]

        @tool
        def web_search_tool(query: str) -> str:
            """
            사용자가 얼굴형, 얼굴톤, 계절별 헤어스타일 추천 외의 요즘 유행하는 헤어스타일 같은 질의 시 해당 정보에 대해 웹 검색을 실행합니다.
            결과는 항상 한국에서 나온 정보들만 사용합니다.
            지금은 2025년도 입니다.
            """
            # return web_search(query)

            search = DuckDuckGoSearchRun()
            res = search.run(query)
            return res

        
        @tool
        def rag_search_tool(face_shape: str|None=None, season: str|None=None, tone: str|None=None):
            """
            사용자가 얼굴형, 얼굴톤, 계절별 헤어스타일 추천 질의 시 vectorDB 에서 검색을 수행합니다.
            """

            return rag_search(face_shape, season, tone)
        
        tools = get_tool_list(hairstyle_recommendation_tool,non_image_recommendation_tool, hairstyle_generation_tool, web_search_tool)

        # Agent 생성
        agent = create_openai_tools_agent(llm, tools, prompt)

        # AgentExecutor 생성
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=30,
            max_execution_time=300,
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
        # 캐시 플래그 초기화 (새 요청 시작 시 이전 상태 리셋)
        self.last_tool_cache_hit = False
        self.cached_final_answer = None
        self.last_tool_params = None

        # 입력에서 이미지 추출
        if 'input' in inputs:
            messages = inputs['input']
            self.last_inputs = messages
            for msg in messages:
                if hasattr(msg, 'content') and isinstance(msg.content, list):
                    if any(isinstance(content, dict) and content.get('type') == 'image_url' for content in msg.content):
                        for content in msg.content:
                            if isinstance(content, dict) and content.get('type') == 'image_url':
                                self.current_image_base64 = content['image_url']['url']
                                print(f"[INFO] 이미지 감지! Base64 길이: {len(self.current_image_base64)}")
                                break
                    else:
                        if self.current_image_base64 is not None:
                            print(f"[INFO] 이전 이미지 유지! Base64 길이: {len(self.current_image_base64)}")

        # Agent 실행
        result = self.agent.invoke(inputs, config, **kwargs)

        # 캐시 히트였으면 Tool에서 설정한 답변 반환
        if self.last_tool_cache_hit and self.cached_final_answer:
            print("[CACHE HIT] 캐시된 최종 답변 사용")
            result['output'] = self.cached_final_answer

        # 최종 답변을 캐시에 저장 (tool이 실행되었고 캐시 히트가 아닌 경우)
        if self.last_tool_params is not None and not self.last_tool_cache_hit:
            final_answer = result.get('output', '')
            if final_answer:
                self.store_cache(final_answer, **self.last_tool_params)
                print(f"[CACHE STORE] 최종 답변 캐시 저장 완료")

        return result


def build_agent(model, client):
    """
    HairstyleAgent 인스턴스를 생성하여 반환
    
    Args:
        model: IdentiFace 모델
        
    Returns:
        HairstyleAgent 인스턴스
    """
    return HairstyleAgent(model, client)