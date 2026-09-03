import logging
import os
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

# ── Supported model registry ───────────────────────────────

SUPPORTED_MODELS = [
    "gpt-5.4-mini",
    "gpt-5.5",
    "claude",
    "gemini-2.5",
    "gemini-3.5",
]

MODEL_DISPLAY_NAMES = {
    "gpt-5.4-mini": "GPT-5.4 Mini",
    "gpt-5.5":      "GPT-5.5",
    "claude":       "Claude Sonnet 4.5",
    "gemini-2.5":   "Gemini 2.5 Flash",
    "gemini-3.5":   "Gemini 3.5 Flash",
}


# ── Public factory function ────────────────────────────────

def create_model_client(model_name: str, max_tokens: int = 400):
    """
    Build and return an AutoGen-compatible model client.

    Args:
        model_name  One of SUPPORTED_MODELS.
        max_tokens  Maximum tokens per LLM response (default 400).

    Returns:
        Configured AutoGen model client for use in AssistantAgent.

    Raises:
        ValueError  If model_name is not in SUPPORTED_MODELS.
        KeyError    If the required API key is missing from .env.
        ImportError If the required autogen-ext extra is not installed.
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Supported: {SUPPORTED_MODELS}"
        )

    builders = {
        "gpt-5.4-mini": _build_openai,
        "gpt-5.5":      _build_openai,
        "claude":       _build_claude,
        "gemini-2.5":   _build_gemini,
        "gemini-3.5":   _build_gemini,
    }

    client = builders[model_name](model_name, max_tokens)
    log.info("Model client created: %s", MODEL_DISPLAY_NAMES[model_name])
    return client


# ── OpenAI builder ─────────────────────────────────────────

def _build_openai(model_name: str, max_tokens: int):
    """
    Build an OpenAI chat client for GPT-5.4-mini or GPT-5.5.

    GPT-5.5 requires max_completion_tokens (not max_tokens).
    Both models require explicit ModelInfo because autogen-ext
    does not recognise these identifiers in its built-in registry.
    """
    try:
        from autogen_ext.models.openai import OpenAIChatCompletionClient
        from autogen_core.models import ModelInfo
    except ImportError:
        raise ImportError(
            "OpenAI client missing. Run: pip install autogen-ext[openai]"
        )

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise KeyError(
            "OPENAI_API_KEY not set in .env. "
            "Obtain a key at https://platform.openai.com/api-keys"
        )

    # GPT-5.5 uses max_completion_tokens; GPT-5.4-mini uses max_tokens.
    # We pass max_completion_tokens for both to stay forward-compatible.
    openai_model_id = {
        "gpt-5.4-mini": "gpt-5.4-mini",
        "gpt-5.5":      "gpt-5.5",
    }[model_name]

    return OpenAIChatCompletionClient(
        model=openai_model_id,
        api_key=api_key,
        max_completion_tokens=max_tokens,   # works for both GPT variants
        model_info=ModelInfo(
            vision=True,
            function_calling=True,
            json_output=True,
            family="gpt",
            structured_output=True,
        ),
    )


# ── Anthropic Claude builder ───────────────────────────────

def _build_claude(model_name: str, max_tokens: int):
    """
    Build an Anthropic Claude Sonnet 4.5 client.

    Uses the native AnthropicChatCompletionClient so the
    correct Anthropic API parameter names are used automatically.
    """
    try:
        from autogen_ext.models.anthropic import AnthropicChatCompletionClient
    except ImportError:
        raise ImportError(
            "Anthropic client missing. "
            "Run: pip install autogen-ext[anthropic]"
        )

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise KeyError(
            "ANTHROPIC_API_KEY not set in .env. "
            "Obtain a key at https://console.anthropic.com/"
        )

    return AnthropicChatCompletionClient(
        model="claude-sonnet-4-5",
        api_key=api_key,
        max_tokens=max_tokens,
    )


# ── Google Gemini builder ──────────────────────────────────

def _build_gemini(model_name: str, max_tokens: int):
    """
    Build a Gemini Flash client (2.5 or 3.5) via the
    OpenAI-compatible endpoint.

    Google exposes an OpenAI-format API at:
        https://generativelanguage.googleapis.com/v1beta/openai/

    ModelInfo is provided explicitly because autogen-ext does not
    include Gemini model identifiers in its built-in registry.
    """
    try:
        from autogen_ext.models.openai import OpenAIChatCompletionClient
        from autogen_core.models import ModelInfo
    except ImportError:
        raise ImportError(
            "OpenAI client missing. Run: pip install autogen-ext[openai]"
        )

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise KeyError(
            "GOOGLE_API_KEY not set in .env. "
            "Obtain a key at https://aistudio.google.com/app/apikey"
        )

    # Map our CLI keys to the actual Google model identifiers.
    gemini_model_id = {
        "gemini-2.5": "gemini-2.5-flash",
        "gemini-3.5": "gemini-3.5-flash",
    }[model_name]

    return OpenAIChatCompletionClient(
        model=gemini_model_id,
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        max_tokens=max_tokens,
        model_info=ModelInfo(
            vision=True,
            function_calling=True,
            json_output=True,
            family="unknown",
            structured_output=True,
        ),
    )