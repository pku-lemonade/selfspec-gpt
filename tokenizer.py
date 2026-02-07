import os
import sentencepiece as spm
import tiktoken
from tiktoken.load import load_tiktoken_bpe
from pathlib import Path
from typing import Dict


class TokenizerInterface:
    def __init__(self, model_path):
        self.model_path = model_path

    def encode(self, text):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def decode(self, tokens):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def bos_id(self):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def eos_id(self):
        raise NotImplementedError("This method should be overridden by subclasses.")


class SentencePieceWrapper(TokenizerInterface):
    def __init__(self, model_path):
        super().__init__(model_path)
        self.processor = spm.SentencePieceProcessor(str(model_path))

    def encode(self, text):
        return self.processor.EncodeAsIds(text)

    def decode(self, tokens):
        return self.processor.DecodeIds(tokens)

    def bos_id(self):
        return self.processor.bos_id()

    def eos_id(self):
        return self.processor.eos_id()


class TiktokenWrapper(TokenizerInterface):
    """
    Tokenizing and encoding/decoding text using the Tiktoken tokenizer.
    """

    special_tokens: Dict[str, int]

    num_reserved_special_tokens = 256

    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # noqa: E501

    def __init__(self, model_path):
        super().__init__(model_path)
        assert os.path.isfile(model_path), str(model_path)
        mergeable_ranks = load_tiktoken_bpe(str(model_path))
        num_base_tokens = len(mergeable_ranks)
        special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|reserved_special_token_2|>",
            "<|reserved_special_token_3|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|reserved_special_token_4|>",
            "<|eot_id|>",  # end of turn
        ] + [
            f"<|reserved_special_token_{i}|>"
            for i in range(5, self.num_reserved_special_tokens - 5)
        ]
        self.special_tokens = {
            token: num_base_tokens + i for i, token in enumerate(special_tokens)
        }
        self.model = tiktoken.Encoding(
            name=Path(model_path).name,
            pat_str=self.pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )
        self._bos_id: int = self.special_tokens["<|begin_of_text|>"]
        self._eos_id: int = self.special_tokens["<|end_of_text|>"]

    def encode(self, text):
        return self.model.encode(text)

    def decode(self, tokens):
        return self.model.decode(tokens)

    def bos_id(self):
        return self._bos_id

    def eos_id(self):
        return self._eos_id


class HFTokenizerWrapper(TokenizerInterface):
    def __init__(self, model_path):
        super().__init__(model_path)
        try:
            from transformers import AutoTokenizer
        except Exception as exc:
            raise RuntimeError(
                "Loading tokenizer.json requires transformers. Install with: pip install transformers"
            ) from exc

        model_path = Path(model_path)
        model_dir = model_path if model_path.is_dir() else model_path.parent
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True, use_fast=True)

        bos = self.tokenizer.bos_token_id
        eos = self.tokenizer.eos_token_id
        if isinstance(eos, (list, tuple)):
            eos = eos[0] if eos else None
        if bos is None:
            bos = eos
        if eos is None:
            eos = bos
        if bos is None:
            bos = 0
            eos = 0
        self._bos_id = int(bos)
        self._eos_id = int(eos)

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=False)

    def bos_id(self):
        return self._bos_id

    def eos_id(self):
        return self._eos_id


def resolve_tokenizer_path(model_or_checkpoint_path):
    model_or_checkpoint_path = Path(model_or_checkpoint_path)
    model_dir = model_or_checkpoint_path.parent if model_or_checkpoint_path.is_file() else model_or_checkpoint_path
    for name in ("tokenizer.model", "tokenizer.json"):
        candidate = model_dir / name
        if candidate.is_file():
            return candidate
    return model_dir / "tokenizer.model"


def get_tokenizer(tokenizer_model_path, model_name):
    """
    Factory function to get the appropriate tokenizer based on model family and tokenizer file.
    """
    tokenizer_model_path = Path(tokenizer_model_path)
    model_name = str(model_name).lower()

    if tokenizer_model_path.name == "tokenizer.json" or "qwen" in model_name:
        return HFTokenizerWrapper(tokenizer_model_path)

    if "llama-3" in model_name or "llama-4" in model_name:
        try:
            return TiktokenWrapper(tokenizer_model_path)
        except Exception:
            return SentencePieceWrapper(tokenizer_model_path)

    try:
        return SentencePieceWrapper(tokenizer_model_path)
    except Exception:
        return TiktokenWrapper(tokenizer_model_path)
