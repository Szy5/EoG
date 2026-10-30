import logging
import random
import re
import json
import time
import requests
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from jinja2 import Template
from openai import OpenAI

logger = logging.getLogger(__name__)

_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "template"
_KGQA_RAG_VERIFIER_SYS_TEMPLATE = _TEMPLATE_DIR / "kgqa_rag_verifier_sys.jinja"
_KGQA_RAG_VERIFIER_USER_TEMPLATE = _TEMPLATE_DIR / "kgqa_rag_verifier_user.jinja"


def _read_template(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def load_kgqa_triple_answer_system_prompt() -> str:
    """Load the structured KGQA triple-answer verifier system prompt."""
    if _KGQA_RAG_VERIFIER_SYS_TEMPLATE.is_file():
        return _read_template(_KGQA_RAG_VERIFIER_SYS_TEMPLATE)
    return (
        "You are a careful knowledge-graph QA assistant. Answer using only the provided "
        "initial entities and KG triples. End with Answer: [\"entity\"]."
    )


def build_kgqa_triple_answer_user_prompt(
    *, question: str, question_entities: str, triples_text: str
) -> str:
    if _KGQA_RAG_VERIFIER_USER_TEMPLATE.is_file():
        return Template(_read_template(_KGQA_RAG_VERIFIER_USER_TEMPLATE)).render(
            question=question,
            question_entities=question_entities,
            triples=triples_text,
        )
    return (
        f"Initial Entities:\n{question_entities}\n\n"
        f"KG TRIPLES:\n{triples_text}\n\n"
        f"Question:\n{question}\n"
    )


def parse_kgqa_triple_answer(raw_answer: str) -> List[str]:
    """Parse the structured answer format used by the KGQA VAL/RAG evaluator."""
    text = str(raw_answer or "").strip()
    answer_match = re.search(r"Answer[:：]?\s*(.*)", text, re.IGNORECASE | re.DOTALL)
    if answer_match:
        text = answer_match.group(1).strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        return []

    if isinstance(decoded, dict):
        decoded = decoded.get("answers")
    if isinstance(decoded, str):
        decoded = [decoded]
    if not isinstance(decoded, list):
        return []

    return [str(answer).strip() for answer in decoded if str(answer).strip()]


@dataclass(frozen=True)
class KGQATripleAnswerEvaluation:
    raw_answer: str
    predicted_answers: List[str]
    hit_at_1: float


def evaluate_kgqa_triple_answer(
    judge: Any,
    *,
    question: str,
    question_entities: str,
    triples_text: str,
    gold_answers: List[str],
) -> KGQATripleAnswerEvaluation:
    """Run the KGQA triple-answer verifier and score it with canonical Hit@1."""
    # Lazy import: top-level import creates a cycle
    # llm_as_a_judge -> reward.__init__ -> outcome_score -> llm_as_a_judge.
    from quarl.reward.score.kgqa_answer_em import compute_hit_at_1

    raw_answer = judge.model_based_kgqa_triple_answer(
        question=question,
        question_entities=question_entities,
        triples_text=triples_text,
        temperature=0.0,
    )
    predicted_answers = parse_kgqa_triple_answer(raw_answer)
    return KGQATripleAnswerEvaluation(
        raw_answer=raw_answer,
        predicted_answers=predicted_answers,
        hit_at_1=compute_hit_at_1(predicted_answers, gold_answers),
    )


class SyncLLMAsAJudge:
    """参考 DirectLLMHandler 实现的同步 Judge (OpenAI SDK 模式)"""

    def __init__(self, base_url: str, api_key: Any, model: str,
                 base_url_2: str = None, api_key_2: Any = None, model_2: str = None):
        """
        Initialize synchronous LLM judge

        Args:
            base_url: Base URL for OpenAI API
            api_key: API key (string or list)
            model: Model name to use
            base_url_2: Optional second base URL for load balancing
            api_key_2: Optional second API key
            model_2: Optional second model name
        """
        # 构建配置列表，每个配置包含 (base_url, api_keys, model)
        self.configs = []
        
        # 第一套配置
        if isinstance(api_key, list):
            api_keys_1 = api_key
        elif isinstance(api_key, str) and "," in api_key:
            api_keys_1 = [k.strip() for k in api_key.split(",")]
        else:
            api_keys_1 = [api_key]
        self.configs.append({
            "base_url": base_url,
            "api_keys": api_keys_1,
            "model": model
        })
        
        # 第二套配置（如果提供）
        if base_url_2 and api_key_2 and model_2:
            if isinstance(api_key_2, list):
                api_keys_2 = api_key_2
            elif isinstance(api_key_2, str) and "," in api_key_2:
                api_keys_2 = [k.strip() for k in api_key_2.split(",")]
            else:
                api_keys_2 = [api_key_2]
            self.configs.append({
                "base_url": base_url_2,
                "api_keys": api_keys_2,
                "model": model_2
            })
            logger.info(f"SyncLLMAsAJudge initialized with {len(self.configs)} model configs for load balancing")
        
        # 兼容旧代码的属性
        self.base_url = base_url
        self.model = model
        self.api_keys = api_keys_1

    def _get_client(self):
        """每次请求随机选择一套配置创建 client"""
        config = random.choice(self.configs)
        api_key = random.choice(config["api_keys"])
        return OpenAI(api_key=api_key, base_url=config["base_url"]), config["model"]
    
    def _get_client_legacy(self):
        """兼容旧的单配置模式"""
        api_key = random.choice(self.api_keys)
        return OpenAI(api_key=api_key, base_url=self.base_url)

    def model_based_match(self, question: str, golden_answer: Any, model_answer: str) -> bool:
        """Model-based answer matching"""
        prompt = f"""
Please determine whether the model's answer is consistent with the reference answer:

Question: {question}
Model Answer: {model_answer}
Reference Answer: {golden_answer}

Evaluation Criteria:
1. The model answer must accurately respond to the question and be consistent with the reference answer in meaning.
2. For numerical questions, the values must be equal or very close.
3. For textual questions, the core meaning must be correct.
4. Differences in wording or language are allowed as long as the core answer is the same.
5. If the model answer includes the correct answer and does not contain conflicting information, it is also considered correct.

Please respond only with "Correct" or "Wrong". Do not provide any additional explanation.
"""
        retry = 0
        max_retry = 3
        while retry < max_retry:
            try:
                client, model = self._get_client()
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a professional judge who evaluates the correctness of answers based on given criteria.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                    max_tokens=2048,
                )

                judgment = response.choices[0].message.content.strip()
                logger.debug(f"LLM judgment result: {judgment}")
                return "correct" in judgment.lower()

            except Exception as e:
                logger.error(f"LLM judgment error (attempt {retry+1}/{max_retry}): {e}")
                retry += 1
                time.sleep(1)
        return False

    def batch_model_based_match(self, evaluations: list[dict]) -> list[bool]:
        results = []
        for eval_data in evaluations:
            results.append(self.model_based_match(
                question=eval_data["question"],
                golden_answer=eval_data["golden_answer"],
                model_answer=eval_data["model_answer"]
            ))
        return results

    def model_based_answer(self, materials, question) -> str:
        """
        Model-based answer generation
        """
        prompt = f"Answer the given question based on the provided materials. You should first conduct very concise reasoning within 200  words, and then directly provide your answer without detailed illustrations after saying 'Answer:'. Materials: {materials}\n Question: {question}\n"
        
        retry = 0
        max_retry = 3
        while retry < max_retry:
            try:
                client, model = self._get_client()
                response_result = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                    max_tokens=2048,
                )
                return response_result.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"LLM answer error (attempt {retry+1}/{max_retry}): {e}")
                retry += 1
                time.sleep(1)
        return ""

    def model_based_kgqa_triple_answer(
        self,
        *,
        question: str,
        question_entities: str,
        triples_text: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        """Answer a KGQA question from structured exploration triples."""
        system_prompt = system_prompt or load_kgqa_triple_answer_system_prompt()
        user_prompt = build_kgqa_triple_answer_user_prompt(
            question=question,
            question_entities=question_entities,
            triples_text=triples_text,
        )
        retry = 0
        max_retry = 3
        while retry < max_retry:
            try:
                client, model = self._get_client()
                response_result = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response_result.choices[0].message.content.strip()
            except Exception as e:
                logger.error(
                    "KGQA triple answer error (attempt %s/%s): %s",
                    retry + 1,
                    max_retry,
                    e,
                )
                retry += 1
                time.sleep(1)
        return ""

    def close(self):
        pass


class APILLMAsAJudge:
    """API-based LLM-as-a-Judge implementation using raw HTTP requests."""

    def __init__(self, base_url: str, api_key: Any, model: str, header_type: str = "openai",
                 base_url_2: str = None, api_key_2: Any = None, model_2: str = None,
                 header_type_2: str = None):
        """
        Initialize API-based LLM judge with optional second configuration for load balancing.

        Args:
            base_url: Base URL for the first API endpoint
            api_key: API key for the first endpoint
            model: Model name for the first endpoint
            header_type: Header type for first endpoint (openai/direct)
            base_url_2: Optional second base URL for load balancing
            api_key_2: Optional second API key
            model_2: Optional second model name
            header_type_2: Optional header type for second endpoint (defaults to header_type)
        """
        self.header_type = header_type
        
        # 构建配置列表
        self.configs = []
        
        # 处理第一套配置
        def build_url(url):
            if url.endswith('/chat/completions'):
                return url
            elif url.endswith('/'):
                return url + "chat/completions"
            else:
                return url + "/chat/completions"
        
        def parse_api_keys(key):
            if isinstance(key, list):
                return key
            elif isinstance(key, str) and "," in key:
                return [k.strip() for k in key.split(",")]
            else:
                return [key]
        
        self.configs.append({
            "url": build_url(base_url),
            "api_keys": parse_api_keys(api_key),
            "model": model,
            "header_type": header_type  # 每套配置独立的 header_type
        })
        
        # 第二套配置（如果提供）
        if base_url_2 and api_key_2 and model_2:
            # 第二套配置的 header_type，如果未指定则使用第一套的
            ht2 = header_type_2 if header_type_2 else header_type
            self.configs.append({
                "url": build_url(base_url_2),
                "api_keys": parse_api_keys(api_key_2),
                "model": model_2,
                "header_type": ht2
            })
            logger.info(f"APILLMAsAJudge initialized with {len(self.configs)} model configs for load balancing")
            logger.info(f"  Config 1: model={model}, header_type={header_type}")
            logger.info(f"  Config 2: model={model_2}, header_type={ht2}")
        
        # 兼容旧代码的属性
        self.url = self.configs[0]["url"]
        self.model = model
        self.api_keys = self.configs[0]["api_keys"]

    @staticmethod
    def _build_headers(api_key: str, header_type: str) -> Dict[str, str]:
        """Build HTTP headers for the judge API request.

        Args:
            header_type: "openai" (default, ``Authorization: Bearer <key>``)
                         or "direct" (``Authorization: <key>``).
        """
        if header_type == "direct":
            return {
                'Authorization': api_key,
                'Content-Type': 'application/json'
            }
        else:  # default: standard OpenAI Bearer token
            return {
                'Authorization': f"Bearer {api_key}",
                'Content-Type': 'application/json'
            }

    def _get_config(self) -> tuple:
        """每次请求随机选择一套配置，返回 (url, headers, model)"""
        config = random.choice(self.configs)
        api_key = random.choice(config["api_keys"])
        headers = self._build_headers(api_key, config["header_type"])
        return config["url"], headers, config["model"]

    def _get_headers(self) -> Dict[str, str]:
        """兼容旧代码的方法"""
        api_key = random.choice(self.api_keys)
        return self._build_headers(api_key, self.header_type)

    def _call_api(self, messages: List[Dict], temperature: float = 0.1, max_tokens: int = 10) -> str:
        retry = 0
        max_retry = 5  # 增加重试次数
        while retry < max_retry:
            try:
                # 每次重试随机选择一套配置
                url, headers, model = self._get_config()
                
                payload = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": False
                }
                
                response = requests.post(
                    url,
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=30.0
                )
                if response.status_code == 429:
                    # 限流错误，等待更长时间后重试
                    wait_time = min(2 ** retry * 2, 60)  # 指数退避，最长60秒
                    logger.warning(f"Rate limited (429), waiting {wait_time}s before retry {retry+1}/{max_retry}")
                    time.sleep(wait_time)
                    retry += 1
                    continue
                if response.status_code != 200:
                    logger.error(f"API Error: {response.status_code} {response.text}")
                    raise Exception(f"HTTP {response.status_code}")

                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    message = result['choices'][0]['message']
                    content = message.get('content') or ''
                    
                    # qwen3 模型可能把答案放在 reasoning_content 里
                    if not content and 'reasoning_content' in message:
                        reasoning = message.get('reasoning_content', '')
                        # 从 reasoning_content 中提取答案
                        if 'Answer:' in reasoning:
                            content = reasoning.split('Answer:')[-1].strip()
                        elif 'answer:' in reasoning:
                            content = reasoning.split('answer:')[-1].strip()
                        else:
                            # 取最后一段作为答案
                            content = reasoning.strip().split('\n')[-1].strip()
                    
                    if content and '<think>' in content and '</think>' in content:
                        content = content.split('</think>')[-1].strip()
                    return content.strip() if content else ""
                return ""
            except Exception as e:
                logger.error(f"API call error (attempt {retry+1}/{max_retry}): {e}")
                retry += 1
                time.sleep(1)
        return ""

    def model_based_match(self, question: str, golden_answer: Any, model_answer: str) -> bool:
        prompt = f"""
Please determine whether the model’s answer is consistent with the reference answer:

Question: {question}
Model Answer: {model_answer}
Reference Answer: {golden_answer}

Evaluation Criteria:
1. The model answer must accurately respond to the question and be consistent with the reference answer in meaning.
2. For numerical questions, the values must be equal or very close.
3. For textual questions, the core meaning must be correct.
4. Differences in wording or language are allowed as long as the core answer is the same.
5. If the model answer includes the correct answer and does not contain conflicting information, it is also considered correct.

Please respond only with "Correct" or "Wrong". Do not provide any additional explanation.
"""
        messages = [
            {"role": "system", "content": "You are a professional judge who evaluates the correctness of answers."},
            {"role": "user", "content": prompt}
        ]
        judgment = self._call_api(messages)
        logger.debug(f"API judgment result: {judgment}")
        return "correct" in judgment.lower()

    def batch_model_based_match(self, evaluations: list[dict]) -> list[bool]:
        return [self.model_based_match(e["question"], e["golden_answer"], e["model_answer"]) for e in evaluations]

    def model_based_answer(self, materials, question) -> str:
        prompt = f"Answer the given question based on the provided materials. You should first conduct very concise reasoning within 200 words, and then directly provide your answer without detailed illustrations after saying 'Answer:'. Materials: {materials}\n Question: {question}\n"
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        return self._call_api(messages, max_tokens=2048)

    def model_based_kgqa_triple_answer(
        self,
        *,
        question: str,
        question_entities: str,
        triples_text: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        """Answer a KGQA question from structured exploration triples."""
        system_prompt = system_prompt or load_kgqa_triple_answer_system_prompt()
        user_prompt = build_kgqa_triple_answer_user_prompt(
            question=question,
            question_entities=question_entities,
            triples_text=triples_text,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self._call_api(messages, temperature=temperature, max_tokens=max_tokens)

    def close(self):
        pass


# Singleton pattern management
_global_judge = None


def get_global_judge(base_url: str, api_key: str, model: str = "gpt-4") -> Any:
    """
    Get global judge instance.
    
    Automatically detects second model configuration from environment variables:
    - JUDGE_BASE_URL_2: Second API base URL
    - JUDGE_API_KEY_2: Second API key
    - JUDGE_MODEL_2: Second model name
    - JUDGE_HEADER_TYPE_2: Second API header type (optional, defaults to JUDGE_HEADER_TYPE)
    
    If all three are set, enables load balancing between two model endpoints.
    """
    global _global_judge
    if _global_judge is None:
        import os
        judge_type = os.getenv("JUDGE_TYPE", "openai").lower()
        
        # 检测第二套配置
        base_url_2 = os.getenv("JUDGE_BASE_URL_2")
        api_key_2 = os.getenv("JUDGE_API_KEY_2")
        model_2 = os.getenv("JUDGE_MODEL_2")
        header_type_2 = os.getenv("JUDGE_HEADER_TYPE_2")  # 第二套的 header type
        
        has_second_config = bool(base_url_2 and api_key_2 and model_2)
        if has_second_config:
            logger.info(f"Detected second model config: model={model_2}, will enable load balancing")
        
        if judge_type == "api":
            header_type = os.getenv("JUDGE_HEADER_TYPE", "openai").lower()
            ht2 = header_type_2.lower() if header_type_2 else None
            _global_judge = APILLMAsAJudge(
                base_url=base_url, api_key=api_key, model=model, header_type=header_type,
                base_url_2=base_url_2, api_key_2=api_key_2, model_2=model_2,
                header_type_2=ht2
            )
        else:
            # 默认或 direct 类型使用 SDK 模式，支持多 Key 随机轮询
            _global_judge = SyncLLMAsAJudge(
                base_url=base_url, api_key=api_key, model=model,
                base_url_2=base_url_2, api_key_2=api_key_2, model_2=model_2
            )
    return _global_judge


def create_sync_llm_judge(base_url: str, api_key: str, model: str = "gpt-4") -> Any:
    """
    Create synchronous LLM judge instance.
    
    Automatically detects second model configuration from environment variables:
    - JUDGE_BASE_URL_2: Second API base URL
    - JUDGE_API_KEY_2: Second API key
    - JUDGE_MODEL_2: Second model name
    - JUDGE_HEADER_TYPE_2: Second API header type (optional)
    """
    import os
    judge_type = os.getenv("JUDGE_TYPE", "openai").lower()
    
    # 检测第二套配置
    base_url_2 = os.getenv("JUDGE_BASE_URL_2")
    api_key_2 = os.getenv("JUDGE_API_KEY_2")
    model_2 = os.getenv("JUDGE_MODEL_2")
    header_type_2 = os.getenv("JUDGE_HEADER_TYPE_2")
    
    if judge_type == "api":
        header_type = os.getenv("JUDGE_HEADER_TYPE", "openai").lower()
        ht2 = header_type_2.lower() if header_type_2 else None
        return APILLMAsAJudge(
            base_url=base_url, api_key=api_key, model=model, header_type=header_type,
            base_url_2=base_url_2, api_key_2=api_key_2, model_2=model_2,
            header_type_2=ht2
        )
    return SyncLLMAsAJudge(
        base_url=base_url, api_key=api_key, model=model,
        base_url_2=base_url_2, api_key_2=api_key_2, model_2=model_2
    )


# Usage example
def example_usage():
    """Usage example"""
    import os

    base_url = os.getenv("JUDGE_BASE_URL")
    model = os.getenv("JUDGE_MODEL")
    judge = create_sync_llm_judge(base_url=base_url, api_key="your-api-key", model=model)

    # Single evaluation
    result = judge.model_based_match(
        question="What is the capital of France?", golden_answer="Paris", model_answer="Paris is the capital city of France"
    )
    print(f"Evaluation result: {result}")

    # Batch evaluation
    evaluations = [
        {"question": "What is 1+1?", "golden_answer": "2", "model_answer": "The answer is 2"},
        {"question": "How many moons does Earth have?", "golden_answer": "1", "model_answer": "Earth has one natural satellite, the Moon"},
    ]

    batch_results = judge.batch_model_based_match(evaluations)
    print(f"Batch evaluation results: {batch_results}")

    # Explicit close (optional)
    judge.close()


if __name__ == "__main__":
    example_usage()
