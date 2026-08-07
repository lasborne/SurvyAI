from __future__ import annotations

from typing import Any, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

from survyai.cloud_api import proxy_llm_chat


class SurvyAIProxyChatModel(BaseChatModel):
    base_url: str
    access_token: str
    device_id: str = ""
    provider: str
    model_name: str
    temperature: float = 0.3
    max_tokens: int = 4096
    proxy_path: str = "/v1/llm/chat"
    bound_tools: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "survyai_proxy"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model_name": self.model_name,
            "proxy_path": self.proxy_path,
        }

    def bind_tools(self, tools: Sequence[dict[str, Any] | BaseTool | Any], **kwargs: Any) -> "SurvyAIProxyChatModel":
        converted: list[dict[str, Any]] = []
        for tool in tools:
            converted.append(convert_to_openai_tool(tool))
        return self.model_copy(update={"bound_tools": converted})

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        _ = stop, run_manager, kwargs
        payload = {
            "provider": self.provider,
            "model": self.model_name,
            "temperature": float(self.temperature),
            "max_tokens": int(self.max_tokens),
            "messages": [self._serialize_message(msg) for msg in messages],
            "tools": list(self.bound_tools or []),
        }
        data = proxy_llm_chat(
            base_url=self.base_url,
            access_token=self.access_token,
            device_id=self.device_id or None,
            proxy_path=self.proxy_path,
            payload=payload,
            timeout_s=180,
        )
        tool_calls = []
        for item in data.get("tool_calls") or []:
            if not isinstance(item, dict):
                continue
            tool_calls.append(
                {
                    "id": item.get("id"),
                    "name": item.get("name"),
                    "args": item.get("args") if isinstance(item.get("args"), dict) else {},
                    "type": "tool_call",
                }
            )
        response_metadata = {
            "provider": data.get("provider"),
            "model_name": data.get("model"),
            "billing": data.get("billing") if isinstance(data.get("billing"), dict) else {},
        }
        usage = self._normalize_usage_metadata(
            data.get("usage") if isinstance(data.get("usage"), dict) else {}
        )
        ai = AIMessage(
            content=data.get("content", ""),
            tool_calls=tool_calls,
            response_metadata=response_metadata,
            usage_metadata=usage,
        )
        return ChatResult(generations=[ChatGeneration(message=ai)])

    @staticmethod
    def _normalize_usage_metadata(raw: dict[str, Any]) -> dict[str, Any]:
        """Return LangChain-compatible AIMessage usage_metadata.

        LangChain Core now requires `total_tokens`.  The SurvyAI proxy also sends
        billing-only fields (e.g. cost_usd) that must not be placed inside
        usage_metadata.
        """
        input_tokens = int(raw.get("input_tokens") or 0)
        output_tokens = int(raw.get("output_tokens") or 0)
        total_tokens = int(raw.get("total_tokens") or (input_tokens + output_tokens))
        out: dict[str, Any] = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }
        cached = int(raw.get("cached_input_tokens") or 0)
        if cached:
            out["input_token_details"] = {"cache_read": cached}
        return out

    def _serialize_message(self, message: BaseMessage) -> dict[str, Any]:
        if isinstance(message, SystemMessage):
            return {"role": "system", "content": message.content}
        if isinstance(message, HumanMessage):
            return {"role": "user", "content": message.content}
        if isinstance(message, ToolMessage):
            payload = {
                "role": "tool",
                "content": message.content,
                "tool_call_id": getattr(message, "tool_call_id", None),
            }
            name = getattr(message, "name", None)
            if name:
                payload["name"] = name
            return payload
        if isinstance(message, AIMessage):
            tool_calls = []
            for tc in message.tool_calls or []:
                if isinstance(tc, dict):
                    name = tc.get("name")
                    args = tc.get("args") if isinstance(tc.get("args"), dict) else {}
                    tid = tc.get("id")
                else:
                    name = getattr(tc, "name", None)
                    raw_args = getattr(tc, "args", None)
                    args = raw_args if isinstance(raw_args, dict) else {}
                    tid = getattr(tc, "id", None)
                if not name:
                    continue
                tool_calls.append({"id": tid, "name": name, "args": args})
            content = message.content
            if content is not None and not isinstance(content, (str, list, dict, int, float, bool)):
                content = str(content)
            return {
                "role": "assistant",
                "content": content if content is not None else "",
                "tool_calls": tool_calls,
            }
        return {"role": "user", "content": getattr(message, "content", str(message))}


__all__ = ["SurvyAIProxyChatModel"]
