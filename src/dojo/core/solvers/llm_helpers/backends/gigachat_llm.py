import os
import requests
import json
import time
from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Tuple, Union
from gigachat import GigaChat
from dotenv import load_dotenv
load_dotenv()

def fetch_and_set_gigachat_token():
    giga = GigaChat(
        credentials=os.environ["GIGACHAT_API_KEY"],
    )
    response = giga.get_token()
    token = response.access_token
    return token


@dataclass
class FunctionSpec:
    name: str
    json_schema: Dict[str, Any]
    description: str

    @property
    def as_openai_tool_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.json_schema,
        }

class GigaChatClient:
    PromptType = Union[str, Dict[str, Any], List[Any]]
    OutputType = Union[str, Dict[str, Any]]

    def __init__(self, client_cfg):
        self.model = client_cfg.model_id
        self.base_url = client_cfg.base_url.rstrip("/")
        self.access_token = fetch_and_set_gigachat_token()


    @property
    def client_content_key(self):
        return "content"

    def _query_client(
        self,
        messages: List[Dict[str, str]],
        model_kwargs: Dict[str, Any] = {},
        json_schema: Optional[str] = None,
        function_name: Optional[str] = None,
        function_description: Optional[str] = None,
    ) -> Tuple[OutputType, Dict[str, Any]]:
        # Prepare function spec if provided (if GigaChat supports it)
        func_spec = None
        if json_schema and function_name and function_description:
            func_spec = FunctionSpec(function_name, json.loads(json_schema), function_description)
            # If GigaChat supports function calling, add to payload here

        payload = {
            "model": self.model,
            "messages": messages,
            **model_kwargs,
        }
        # If function calling is supported, add to payload
        if func_spec is not None:
            payload["functions"] = [func_spec.as_openai_tool_dict]
            payload["function_call"] = 'auto'

        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }

        start_time = time.monotonic()
        response = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=60,
        )
        response.raise_for_status()
        latency = time.monotonic() - start_time
        data = response.json()

        # Parse output and usage stats
        output = data["choices"][0]["message"]["content"]
        usage_stats = data.get("usage", {})
        usage_stats["latency"] = latency

        # If function calling, try to extract function call result
        if func_spec is not None and "functions" in payload:
            function_call = data["choices"][0]["message"].get("function_call")
            if function_call:
                try:
                    output = json.loads(function_call["arguments"])
                except Exception:
                    pass  # fallback to text

        return output, usage_stats

    def query(
        self,
        messages: List[Dict[str, str]],
        json_schema: Optional[str] = None,
        function_name: Optional[str] = None,
        function_description: Optional[str] = None,
        **model_kwargs,
    ) -> OutputType:
        return self._query_client(
            messages=messages,
            model_kwargs=model_kwargs,
            json_schema=json_schema,
            function_name=function_name,
            function_description=function_description,
        )