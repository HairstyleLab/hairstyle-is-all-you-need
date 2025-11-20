import os
import torch
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace, HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from typing import List, Optional
from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from pydantic import ConfigDict  # pydantic v2


def load_embedding_model(model_name, device='cpu'):
    if 'text-embedding-3' in model_name:
        embedding_model = OpenAIEmbeddings(model_name=model_name)
    else:
        embedding_model = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'device': device}, encode_kwargs={'normalize_embeddings':True})

    return embedding_model

def load_reranker_model(model_name, device='cpu'):
    reranker_model = HuggingFaceCrossEncoder(model_name=model_name, model_kwargs={'device': device})

    return reranker_model

def load_ollama(model_name, temperature=0.1):
    model = ChatOllama(model=model_name, temperature=temperature)

    return model

def load_openai(model_name='gpt-4o-mini', temperature=0.1):
    model = ChatOpenAI(model=model_name, temperature=temperature)

    return model

def use_endpoint(model_name, token):
    endpoint = HuggingFaceEndpoint(
        repo_id=model_name,
        task='text-generation',
        max_new_tokens=1024,
        huggingfacehub_api_token=token,
    )

    model = ChatHuggingFace(llm=endpoint, verbose=True)

    return model

def load_hf(model_name):
    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        model_name,
        dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_name)
    qwen3vl_chat = Qwen3VLChat(model=model, processor=processor)

    return qwen3vl_chat

class Qwen3VLChat(BaseChatModel):
    model: any          # <- 구체 타입 말고 Any
    processor: any      # <- 여기도 Any

    # arbitrary 타입 허용 (보통 LangChain에서 이미 처리돼 있지만 안전하게 한 번 더)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def _llm_type(self) -> str:
        return "qwen3-vl"

    def _convert_messages_to_qwen_format(self, messages: List[BaseMessage]):
        """
        LangChain 메시지 리스트 -> Qwen chat_template용 messages 포맷으로 변환
        """
        qwen_messages = []
        user_content = []

        for m in messages:
            if isinstance(m, HumanMessage):
                # content가 이미 멀티모달 리스트인 경우
                if isinstance(m.content, list):
                    user_content.extend(m.content)
                # 순수 텍스트만 있는 경우
                elif isinstance(m.content, str):
                    user_content.append({"type": "text", "text": m.content})

        if not user_content:
            user_content.append({"type": "text", "text": "Hello"})

        qwen_messages.append({
            "role": "user",
            "content": user_content,
        })
        return qwen_messages

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> ChatResult:
        # LangChain messages -> Qwen chat_template 포맷
        qwen_messages = self._convert_messages_to_qwen_format(messages)

        # Qwen processor 적용
        inputs = self.processor.apply_chat_template(
            qwen_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        max_new_tokens = kwargs.get("max_new_tokens", 256)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
            )

        # 프롬프트 부분 잘라내기
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        ai_msg = AIMessage(content=output_text)
        return ChatResult(generations=[ChatGeneration(message=ai_msg)])
