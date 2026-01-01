import json
import ollama
from typing import List, Dict
from abc import ABC, abstractmethod


class BaseLLMProvider(ABC):
    @abstractmethod
    def generate_queries(self, prompt: str, num_queries: int = 3) -> List[str]:
        pass


class OllamaProvider(BaseLLMProvider):
    def __init__(self, model: str = "qwen3:8b"):
        try:
            self.ollama = ollama
        except ImportError:
            raise ImportError("ollama 설치 필요")

        self.model = model
        self._check_model()

    def _check_model(self):
        try:
            models = self.ollama.list()
            available_models = [m['name'] for m in models.get('models', [])]

            model_exists = any(
                self.model == m or self.model in m
                for m in available_models
            )

            if not model_exists:
                print(f"경고: 모델 '{self.model}'을 찾을 수 없습니다.")
                print(f"사용 가능한 모델: {available_models}")
                print(f"모델을 자동으로 다운로드합니다...")
                self.ollama.pull(self.model)
        except Exception as e:
            print(f"모델 확인 중 오류: {e}")

    def generate_queries(self, prompt: str, num_queries: int = 1) -> List[str]:
        prompt_with_format = f"{prompt}\n\n중요: 반드시 JSON 형식으로만 응답하세요. 다른 설명은 포함하지 마세요."

        try:
            response = self.ollama.generate(
                model=self.model,
                prompt=prompt_with_format,
                format='json',
                options={
                    'temperature': 0.7,
                }
            )

            response_text = response.get('response', '{}')

            try:
                parsed = json.loads(response_text)
                if isinstance(parsed, dict):
                    queries = parsed.get('queries', parsed.get('keywords', []))
                elif isinstance(parsed, list):
                    queries = parsed
                else:
                    queries = []

                return queries[:num_queries]
            except json.JSONDecodeError:
                print(f"JSON 파싱 실패: {response_text[:100]}")
                return []

        except Exception as e:
            print(f"Ollama 호출 실패: {e}")
            return []

class OpenAIProvider(BaseLLMProvider):
    def __init__(self, model: str = "gpt-4o-mini"):
        try:
            from openai import OpenAI
            import os
        except ImportError:
            raise ImportError("openai 설치 필요:")

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate_queries(self, prompt: str, num_queries: int = 3) -> List[str]:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "당신은 검색 쿼리 생성 전문가입니다."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.7
            )

            response_text = response.choices[0].message.content

            try:
                parsed = json.loads(response_text)
                if isinstance(parsed, dict):
                    queries = parsed.get('queries', parsed.get('keywords', []))
                elif isinstance(parsed, list):
                    queries = parsed
                else:
                    queries = []

                return queries[:num_queries]

            except json.JSONDecodeError:
                print(f"JSON 파싱 실패: {response_text[:100]}")
                return []

        except Exception as e:
            print(f"OpenAI 호출 실패: {e}")
            return []


def get_llm_provider(provider_name: str = "ollama", **kwargs) -> BaseLLMProvider:
    providers = {
        'ollama': OllamaProvider,
        'openai': OpenAIProvider
    }

    if provider_name not in providers:
        raise ValueError(
            f"지원하지 않는 provider: {provider_name}\n"
            f"사용 가능한 provider: {list(providers.keys())}"
        )

    return providers[provider_name](**kwargs)


if __name__ == "__main__":
    print("1. Ollama 테스트...")
    try:
        ollama = get_llm_provider("ollama", model="qwen3:8b")
        queries = ollama.generate_queries(
            '{"queries": ["쿼리1", "쿼리2", "쿼리3"]} 형식으로 헤어스타일 검색 쿼리 1개 생성',
            num_queries=1
        )
        print(f"Ollama: {queries}")
    except Exception as e:
        print(f"Ollama: {e}")
