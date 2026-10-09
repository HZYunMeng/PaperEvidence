"""Chat Completions compatible transport for configured third-party services."""
import json
import urllib.error
import urllib.request
from urllib.parse import urlparse

from .answering import messages_for


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class CompatibleGenerator:
    provider = "compatible-api"

    def __init__(self, settings, opener=None):
        parsed = urlparse(settings.base_url)
        if parsed.scheme != "https" or not parsed.hostname:
            raise ValueError("云端 API base_url 必须为 HTTPS 服务地址。")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("API 地址不能包含凭证、查询参数或 fragment。")
        if not settings.model.strip() or not settings.api_key.strip():
            raise ValueError("请在 config.toml 设置模型与地址，并通过环境变量配置 API Key。")
        if "\n" in settings.api_key or "\r" in settings.api_key:
            raise ValueError("API Key 格式无效。")
        if not 1 <= settings.max_tokens <= 4096 or not 1 <= settings.timeout <= 120:
            raise ValueError("max_tokens 必须为 1–4096，timeout 必须为 1–120 秒。")
        if not 100 <= settings.max_evidence_chars <= 64000 or type(settings.json_mode) is not bool:
            raise ValueError("证据预算或 JSON 模式配置无效。")
        self.settings, self.endpoint = settings, settings.base_url.rstrip("/") + "/chat/completions"
        self.opener = opener or urllib.request.build_opener(NoRedirect())
        self.last_usage = {}
        self.metadata = {"provider":self.provider,"model":settings.model,
                         "host":parsed.hostname,"max_tokens":settings.max_tokens,
                         "max_evidence_chars":settings.max_evidence_chars,
                         "temperature":0,"json_mode":settings.json_mode}
        if settings.enable_thinking is not None:
            self.metadata["enable_thinking"] = settings.enable_thinking

    def __call__(self, question, hits):
        # Preserve complete retrieved excerpts rather than silently chopping a table row.
        if sum(len(h.chunk.text) for h in hits) > self.settings.max_evidence_chars:
            raise RuntimeError("检索证据超过本次字符预算，请减少检索片段数量或调整配置。")
        payload = {"model":self.settings.model,"messages":messages_for(question,hits),
                   "temperature":0,"max_tokens":self.settings.max_tokens,"stream":False}
        if self.settings.json_mode:
            payload["response_format"] = {"type":"json_object"}
        if self.settings.enable_thinking is not None:
            payload["enable_thinking"] = self.settings.enable_thinking
        request = urllib.request.Request(self.endpoint, data=json.dumps(payload,ensure_ascii=False).encode("utf-8"),
            headers={"Authorization":"Bearer " + self.settings.api_key,"Content-Type":"application/json"},method="POST")
        self.last_usage = {}
        try:
            with self.opener.open(request, timeout=self.settings.timeout) as response:
                data = json.load(response)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"模型服务返回 HTTP {exc.code}；请检查地址、模型、权限和额度。未自动重试。") from None
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError("模型请求连接失败或超时；未自动重试。") from None
        if not isinstance(data, dict) or not isinstance(data.get("choices"), list) or not data["choices"]:
            raise ValueError("模型服务缺少 choices 响应。")
        usage = data.get("usage") or {}
        if isinstance(usage, dict):
            self.last_usage = {k:v for k,v in usage.items() if k in {
                "prompt_tokens","completion_tokens","total_tokens"} and type(v) is int}
        choice = data["choices"][0]
        if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
            raise ValueError("模型服务返回了无效的 message 格式。")
        if choice.get("finish_reason") == "length":
            raise ValueError("模型输出达到 token 限制，未完成的回答不予展示。")
        content = choice.get("message", {}).get("content")
        if not isinstance(content, str):
            raise ValueError("模型服务未返回文本回答。")
        return json.loads(content)
