"""
Provider abstraction over Anthropic, OpenAI, and Google SDKs.

generate(provider, model, system, user, temperature) -> {text, input_tokens, output_tokens}

Adding a new model within an existing family requires only a config entry.
"""
import logging
import os
import time

logger = logging.getLogger(__name__)

_anthropic_client = None
_openai_client = None
_google_client = None


def _get_anthropic():
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic
        _anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _anthropic_client


def _get_openai():
    global _openai_client
    if _openai_client is None:
        import openai
        _openai_client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _openai_client


def _get_google():
    global _google_client
    if _google_client is None:
        from google import genai
        _google_client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    return _google_client


def _call_anthropic(model: str, system: str, user: str, temperature: float) -> dict:
    import anthropic
    client = _get_anthropic()
    for attempt in range(3):
        try:
            msg = client.messages.create(
                model=model,
                max_tokens=2048,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return {
                "text": msg.content[0].text,
                "input_tokens": msg.usage.input_tokens,
                "output_tokens": msg.usage.output_tokens,
            }
        except anthropic.RateLimitError:
            if attempt == 2:
                raise
            time.sleep(5 * (2 ** attempt))
        except anthropic.APIStatusError as e:
            if e.status_code >= 500 and attempt < 2:
                time.sleep(5 * (2 ** attempt))
            else:
                raise


def _call_openai(model: str, system: str, user: str, temperature: float) -> dict:
    import openai
    client = _get_openai()
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return {
                "text": resp.choices[0].message.content,
                "input_tokens": resp.usage.prompt_tokens,
                "output_tokens": resp.usage.completion_tokens,
            }
        except openai.RateLimitError:
            if attempt == 2:
                raise
            time.sleep(5 * (2 ** attempt))
        except openai.APIStatusError as e:
            if e.status_code >= 500 and attempt < 2:
                time.sleep(5 * (2 ** attempt))
            else:
                raise


def _call_google(model: str, system: str, user: str, temperature: float) -> dict:
    from google import genai
    from google.genai import types
    client = _get_google()
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=model,
                contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=temperature,
                ),
            )
            usage = response.usage_metadata
            return {
                "text": response.text,
                "input_tokens": usage.prompt_token_count or 0,
                "output_tokens": usage.candidates_token_count or 0,
            }
        except Exception as e:
            msg = str(e).lower()
            is_transient = any(w in msg for w in ("quota", "rate", "500", "503", "unavailable"))
            if is_transient and attempt < 2:
                time.sleep(5 * (2 ** attempt))
            else:
                raise


_PROVIDERS = {
    "anthropic": _call_anthropic,
    "openai": _call_openai,
    "google": _call_google,
}


def generate(provider: str, model: str, system: str, user: str, temperature: float) -> dict:
    """Call a model and return {text, input_tokens, output_tokens}."""
    fn = _PROVIDERS.get(provider)
    if fn is None:
        raise ValueError(f"Unknown provider: {provider!r}. Must be one of: {list(_PROVIDERS)}")
    return fn(model, system, user, temperature)
