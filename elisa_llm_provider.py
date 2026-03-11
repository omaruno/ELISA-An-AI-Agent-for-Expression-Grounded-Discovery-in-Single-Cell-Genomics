#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ELISA – LLM Provider Abstraction (v3 — Groq, Gemini, OpenAI, Claude)
======================================================================
Supports:
  - groq    → Groq Cloud   (Llama 3.1 8B, free: 500K tokens/day)
  - gemini  → Google AI    (Gemini 2.5 Flash, paid: ~$0.15/1M input)
  - openai  → OpenAI       (GPT-4o-mini, paid: ~$0.15/1M input)
  - claude  → Anthropic    (Claude Sonnet 4, paid: ~$3/1M input)

Groq, Gemini, and OpenAI use the OpenAI-compatible chat.completions API.
Claude uses the native Anthropic SDK (pip install anthropic).

Configuration (environment variables):
  LLM_PROVIDER       = "groq" | "gemini" | "openai" | "claude"  (default: "groq")
  GROQ_API_KEY        = your Groq key           (required if provider=groq)
  GEMINI_API_KEY      = your Google AI key       (required if provider=gemini)
  OPENAI_API_KEY      = your OpenAI key          (required if provider=openai)
  ANTHROPIC_API_KEY   = your Anthropic key       (required if provider=claude)
  LLM_MODEL           = override model name      (optional)
  LLM_MAX_SPEND_EUR   = max spending in EUR      (default: 1.0)

Safety:
  - Built-in spending estimator that REFUSES to make calls once the
    estimated cost exceeds LLM_MAX_SPEND_EUR (default €1.00)
  - Auto-retry with exponential backoff on 429 errors
  - This is a HARD cap in code — independent of any cloud budgets
"""

import os
import sys
import time
import threading

# ── Provider detection ─────────────────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower().strip()

# Provider configs
_PROVIDERS = {
    "groq": {
        "env_key": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "llama-3.1-8b-instant",
        "label": "Groq (Llama 3.1 8B)",
        "api_style": "openai",          # uses OpenAI-compatible SDK
        # Groq free tier: no cost
        "cost_per_1m_input": 0.0,
        "cost_per_1m_output": 0.0,
    },
    "gemini": {
        "env_key": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "default_model": "gemini-2.5-flash",
        "label": "Google AI Studio (Gemini 2.5 Flash)",
        "api_style": "openai",
        # Overestimates for safety
        "cost_per_1m_input": 0.20,       # real: ~$0.15
        "cost_per_1m_output": 0.80,      # real: ~$0.60
    },
    "openai": {
        "env_key": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
        "label": "OpenAI (GPT-4o-mini)",
        "api_style": "openai",
        # GPT-4o-mini pricing (overestimates for safety)
        "cost_per_1m_input": 0.20,       # real: ~$0.15
        "cost_per_1m_output": 0.80,      # real: ~$0.60
    },
    "claude": {
        "env_key": "ANTHROPIC_API_KEY",
        "base_url": None,                # uses native Anthropic SDK
        "default_model": "claude-sonnet-4-20250514",
        "label": "Anthropic (Claude Sonnet 4)",
        "api_style": "anthropic",        # uses Anthropic SDK
        # Claude Sonnet 4 pricing (overestimates for safety)
        "cost_per_1m_input": 4.00,       # real: ~$3.00
        "cost_per_1m_output": 20.00,     # real: ~$15.00
    },
}


# ══════════════════════════════════════════════════════════════
# SPENDING TRACKER — hard cap in code
# ══════════════════════════════════════════════════════════════
class SpendingTracker:
    """
    Tracks estimated spending and REFUSES calls when limit is reached.
    Uses OVERESTIMATED costs to be safe. Thread-safe.
    """
    def __init__(self, max_spend_eur=1.0):
        self.max_spend_eur = max_spend_eur
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_calls = 0
        self.estimated_cost_usd = 0.0
        self._lock = threading.Lock()
        # 1 EUR ≈ 1.05 USD (conservative — if EUR is worth more, we stop earlier)
        self.eur_to_usd = 1.05

    def check_budget(self):
        """Raise RuntimeError if estimated cost exceeds limit."""
        max_usd = self.max_spend_eur * self.eur_to_usd
        if self.estimated_cost_usd >= max_usd:
            raise RuntimeError(
                f"[SPENDING CAP] Estimated cost ${self.estimated_cost_usd:.4f} "
                f"has reached limit of €{self.max_spend_eur:.2f} "
                f"(≈${max_usd:.2f}). Refusing further LLM calls. "
                f"Total calls: {self.total_calls}, "
                f"Input tokens: {self.total_input_tokens}, "
                f"Output tokens: {self.total_output_tokens}. "
                f"Increase LLM_MAX_SPEND_EUR to allow more."
            )

    def record(self, input_tokens, output_tokens, provider_info):
        """Record token usage and update cost estimate."""
        with self._lock:
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_calls += 1
            cost_in = (input_tokens / 1_000_000) * provider_info["cost_per_1m_input"]
            cost_out = (output_tokens / 1_000_000) * provider_info["cost_per_1m_output"]
            self.estimated_cost_usd += cost_in + cost_out

    def summary(self):
        return (
            f"[SPENDING] Calls: {self.total_calls} | "
            f"Tokens: {self.total_input_tokens} in / {self.total_output_tokens} out | "
            f"Est. cost: ${self.estimated_cost_usd:.4f} "
            f"(limit: €{self.max_spend_eur:.2f})"
        )


# Global tracker
_MAX_SPEND = float(os.getenv("LLM_MAX_SPEND_EUR", "1.0"))
_tracker = SpendingTracker(max_spend_eur=_MAX_SPEND)


def get_spending_summary():
    """Get current spending summary string."""
    return _tracker.summary()


# ══════════════════════════════════════════════════════════════
# PROVIDER SETUP
# ══════════════════════════════════════════════════════════════

def get_provider_info():
    if LLM_PROVIDER not in _PROVIDERS:
        supported = ", ".join(_PROVIDERS.keys())
        raise RuntimeError(f"Unknown LLM_PROVIDER='{LLM_PROVIDER}'. Supported: {supported}")
    return _PROVIDERS[LLM_PROVIDER]


def get_model_name():
    override = os.getenv("LLM_MODEL", "").strip()
    if override:
        return override
    return get_provider_info()["default_model"]


def get_llm_client():
    """
    Return an API client for the configured provider.

    - For groq/gemini/openai: returns an OpenAI-compatible client
    - For claude: returns an anthropic.Anthropic client
    """
    info = get_provider_info()
    key = os.getenv(info["env_key"])
    if not key:
        raise RuntimeError(f"{info['env_key']} not set. Export it or switch provider.")

    # ── Anthropic (Claude) — native SDK ──
    if info["api_style"] == "anthropic":
        try:
            import anthropic
        except ImportError:
            raise RuntimeError(
                "Anthropic SDK not installed. Run: pip install anthropic"
            )
        client = anthropic.Anthropic(api_key=key)
        print(f"[LLM] Provider : {info['label']}")
        print(f"[LLM] Model    : {get_model_name()}")
        print(f"[LLM] Spend cap: EUR {_MAX_SPEND:.2f} (set LLM_MAX_SPEND_EUR to change)")
        return client

    # ── OpenAI-compatible providers (groq, gemini, openai) ──
    try:
        from openai import OpenAI
    except ImportError:
        # Fallback: try Groq native SDK if provider is groq
        if LLM_PROVIDER == "groq":
            try:
                from groq import Groq
                print(f"[LLM] Using Groq SDK directly (openai package not found)")
                return Groq(api_key=key)
            except ImportError:
                pass
        raise RuntimeError(
            "Please install the openai package: pip install openai"
        )

    client = OpenAI(api_key=key, base_url=info["base_url"])

    print(f"[LLM] Provider : {info['label']}")
    print(f"[LLM] Model    : {get_model_name()}")
    if info["cost_per_1m_input"] > 0:
        print(f"[LLM] Spend cap: EUR {_MAX_SPEND:.2f} (set LLM_MAX_SPEND_EUR to change)")
    else:
        print(f"[LLM] Cost     : FREE (no spending cap needed)")
    return client


# ══════════════════════════════════════════════════════════════
# ASK LLM — with retry + spending cap
# ══════════════════════════════════════════════════════════════

def _ask_anthropic(client, system_prompt, user_prompt, model, info):
    """Call the Anthropic Messages API (Claude)."""
    res = client.messages.create(
        model=model,
        max_tokens=1024,
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    # Extract text from content blocks
    text = ""
    for block in res.content:
        if hasattr(block, "text"):
            text += block.text

    # Track spending from usage
    _tracker.record(
        input_tokens=res.usage.input_tokens,
        output_tokens=res.usage.output_tokens,
        provider_info=info,
    )
    return text.strip()


def _ask_openai_compat(client, system_prompt, user_prompt, model, info):
    """Call an OpenAI-compatible chat completions API (Groq, Gemini, OpenAI)."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    res = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
    )
    # Track spending
    usage = getattr(res, 'usage', None)
    if usage:
        _tracker.record(
            input_tokens=getattr(usage, 'prompt_tokens', 0),
            output_tokens=getattr(usage, 'completion_tokens', 0),
            provider_info=info,
        )
    else:
        # Estimate if no usage info returned
        est_in = len(system_prompt + user_prompt) // 4
        est_out = len(res.choices[0].message.content) // 4
        _tracker.record(est_in, est_out, info)

    return res.choices[0].message.content.strip()


def ask_llm(client, system_prompt, user_prompt, max_chars=18000,
            max_retries=5, initial_wait=10):
    """
    Send a prompt, return text. Includes:
    - Prompt trimming
    - Spending cap check (BEFORE each call)
    - Auto-retry with exponential backoff on 429 / rate-limit errors
    - Token tracking after each call
    - Automatic dispatch to Anthropic or OpenAI-compatible API
    """
    info = get_provider_info()

    # Check budget BEFORE making the call
    _tracker.check_budget()

    # Trim prompt
    if len(user_prompt) > max_chars:
        user_prompt = user_prompt[:max_chars] + "\n\n[... truncated for length ...]"

    model = get_model_name()

    # Select the right call function
    if info["api_style"] == "anthropic":
        call_fn = _ask_anthropic
    else:
        call_fn = _ask_openai_compat

    # Retry loop with exponential backoff
    last_error = None
    wait = initial_wait
    for attempt in range(max_retries):
        try:
            result = call_fn(client, system_prompt, user_prompt, model, info)

            # Print spending every 20 calls
            if _tracker.total_calls % 20 == 0:
                print(f"  {_tracker.summary()}")

            return result

        except Exception as e:
            error_str = str(e)
            last_error = e

            # Check if it's a rate limit error
            is_rate_limit = any(token in error_str for token in [
                "429", "RESOURCE_EXHAUSTED", "overloaded_error",
            ]) or any(token in error_str.lower() for token in [
                "rate", "quota", "too many requests", "overloaded",
            ])

            if is_rate_limit:
                # Try to extract retry delay from error message
                retry_after = wait
                if "retry in" in error_str.lower() or "retry-after" in error_str.lower():
                    try:
                        import re
                        m = re.search(r'retry[- ](?:in|after)[:\s]*(\d+)', error_str.lower())
                        if m:
                            retry_after = max(int(m.group(1)) + 2, wait)
                    except Exception:
                        pass

                if attempt < max_retries - 1:
                    print(f"  [RATE LIMIT] Attempt {attempt+1}/{max_retries}. "
                          f"Waiting {retry_after}s... ({_tracker.summary()})")
                    time.sleep(retry_after)
                    wait = min(wait * 2, 120)  # exponential backoff, max 2 min
                    continue
                else:
                    raise RuntimeError(
                        f"[LLM ERROR] Rate limit after {max_retries} retries: {e}"
                    )
            else:
                # Non-rate-limit error: don't retry
                raise

    raise RuntimeError(f"[LLM ERROR] Failed after {max_retries} attempts: {last_error}")


# ══════════════════════════════════════════════════════════════
# QUICK SELF-TEST
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("ELISA LLM Provider — Self-Test")
    print("=" * 60)
    print(f"Provider  : {LLM_PROVIDER}")
    print(f"Model     : {get_model_name()}")
    print(f"Spend cap : EUR {_MAX_SPEND:.2f}")
    print(f"API style : {get_provider_info()['api_style']}")

    info = get_provider_info()
    key = os.getenv(info["env_key"])
    print(f"API key   : {'SET' if key else 'NOT SET'} ({info['env_key']})")

    print()
    print("Supported providers:")
    for name, cfg in _PROVIDERS.items():
        k = os.getenv(cfg["env_key"])
        status = "KEY SET" if k else "no key"
        cost = "FREE" if cfg["cost_per_1m_input"] == 0 else f"${cfg['cost_per_1m_input']:.2f}/1M in"
        print(f"  {name:<10} {cfg['label']:<40} [{status}] ({cost})")

    if key:
        print(f"\nRunning test with {info['label']}...")
        client = get_llm_client()
        reply = ask_llm(client, "You are a helpful assistant.", "Say hello in one sentence.")
        print(f"Response  : {reply}")
        print(get_spending_summary())
    else:
        print(f"\nSet {info['env_key']} to run the test.")
        print(f"Example:  export {info['env_key']}=your-key-here")
