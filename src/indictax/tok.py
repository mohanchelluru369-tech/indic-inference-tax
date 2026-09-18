"""Uniform access to tokenizers.

Spec formats accepted by `load_tokenizer`:
    org/repo            Hugging Face repo that ships a tokenizer.json
    hf:org/repo         same, explicit
    file:path.json      a local tokenizer.json
    tiktoken:o200k_base an OpenAI encoding (needs `pip install indictax[openai-tok]`)

Gated repos (Llama, Gemma) need HF_TOKEN in the environment, or use an
ungated mirror of the same tokenizer.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Callable

_BYTE_TOKEN = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")


@dataclass
class TokenizerAdapter:
    name: str
    vocab_size: int
    _encode: Callable[[str], list[int]]
    _is_fragment: Callable[[int], bool]

    def encode(self, text: str) -> list[int]:
        return self._encode(text)

    def is_fragment(self, token_id: int) -> bool:
        """True when the token is not valid text on its own: a lone byte or a
        partial UTF-8 sequence. A high fragment rate means the tokenizer never
        learned merges for this script and is spelling it out byte by byte."""
        return self._is_fragment(token_id)


def _from_hf_tokenizer(name: str, tk) -> TokenizerAdapter:
    cache: dict[int, bool] = {}

    def is_fragment(tid: int) -> bool:
        if tid not in cache:
            raw = tk.id_to_token(tid) or ""
            text = tk.decode([tid], skip_special_tokens=False)
            cache[tid] = bool(_BYTE_TOKEN.match(raw)) or "�" in text
        return cache[tid]

    return TokenizerAdapter(
        name=name,
        vocab_size=tk.get_vocab_size(with_added_tokens=True),
        _encode=lambda s: tk.encode(s, add_special_tokens=False).ids,
        _is_fragment=is_fragment,
    )


def _from_tiktoken(name: str, enc) -> TokenizerAdapter:
    cache: dict[int, bool] = {}

    def is_fragment(tid: int) -> bool:
        if tid not in cache:
            try:
                enc.decode_single_token_bytes(tid).decode("utf-8")
                cache[tid] = False
            except UnicodeDecodeError:
                cache[tid] = True
        return cache[tid]

    return TokenizerAdapter(
        name=name,
        vocab_size=enc.n_vocab,
        _encode=lambda s: enc.encode(s, disallowed_special=()),
        _is_fragment=is_fragment,
    )


def load_tokenizer(spec: str) -> TokenizerAdapter:
    if spec.startswith("tiktoken:"):
        try:
            import tiktoken
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("tiktoken not installed: pip install 'indictax[openai-tok]'") from e
        return _from_tiktoken(spec, tiktoken.get_encoding(spec.split(":", 1)[1]))

    from tokenizers import Tokenizer

    if spec.startswith("file:"):
        path = spec.split(":", 1)[1]
        return _from_hf_tokenizer(f"file:{os.path.basename(path)}", Tokenizer.from_file(path))

    repo = spec.split(":", 1)[1] if spec.startswith("hf:") else spec
    token = os.environ.get("HF_TOKEN") or None
    return _from_hf_tokenizer(repo, Tokenizer.from_pretrained(repo, token=token))
