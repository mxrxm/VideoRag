import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import BaseLLM


class HuggingFaceLLM(BaseLLM):
    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        torch_dtype="auto",
        max_new_tokens: int = 1000,
        temperature: float = 0.7,
        top_p: float = 0.95,
        is_chat_model: bool = True,
        system_prompt: str = "You are a helpful assistant.",
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.is_chat_model = is_chat_model
        self.system_prompt = system_prompt

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Some models don't have pad_token -> use eos_token
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device,  # "cpu", "auto", etc.
        )

    def _build_chat_prompt(self, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        chat_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return chat_prompt
    
    def generate(self, prompt: str, **kwargs) -> str:
        if self.is_chat_model:
            prompt_text = self._build_chat_prompt(prompt)
        else:
            prompt_text = prompt

        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        output = self.model.generate(
            **inputs,
            max_new_tokens=kwargs.get("max_new_tokens", self.max_new_tokens),
            temperature=kwargs.get("temperature", self.temperature),
            top_p=kwargs.get("top_p", self.top_p),
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
        )

        
        generated = output.sequences[0]
        input_length = inputs["input_ids"].shape[1]
        new_tokens = generated[input_length:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return text.strip()


