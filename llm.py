from typing import Union, Literal
from langchain.chat_models import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain_anthropic import ChatAnthropic
from langchain import OpenAI
from langchain.schema import (
    HumanMessage
)
import torch
from transformers import BitsAndBytesConfig, LlamaTokenizer, LlamaForCausalLM


class AnyOpenAILLM:
    def __init__(self, *args, **kwargs):
        # Determine model type from the kwargs
        model_name = kwargs.get('model_name', 'gpt-3.5-turbo') 
        if model_name.split('-')[0] == 'text':
            self.model = OpenAI(*args, **kwargs)
            self.model_type = 'completion'
        else:
            self.model = ChatOpenAI(*args, **kwargs)
            self.model_type = 'chat'
    
    def __call__(self, prompt: str):
        if self.model_type == 'completion':
            return self.model(prompt)
        else:
            return self.model(
                [
                    HumanMessage(
                        content=prompt,
                    )
                ]
            ).content
        
class AnyAnthropicLLM:
    def __init__(self, *args, **kwargs):
        self.model = ChatAnthropic(*args, **kwargs)

    def __call__(self, prompt: str):
        return self.model(
            [
                HumanMessage(
                    content=prompt,
                )
            ]
        ).content
        
class AnyMistralAILLM:
    def __init__(self, *args, **kwargs):
        self.model = ChatMistralAI(*args, **kwargs)

    def __call__(self, prompt: str):
        return self.model(
            [
                HumanMessage(
                    content=prompt,
                )
            ]
        ).content
        
class AnyLlamaAILLM:
    def __init__(self, *args, **kwargs):
        model_name = kwargs.get('model_name', 'lmsys/vicuna-7b-v1.5')
        quantization_config = BitsAndBytesConfig(load_in_8bit = True)
        self.model = LlamaForCausalLM.from_pretrained(
            model_name,
            device_map = "auto",
            torch_dtype = torch.float16,
            quantization_config=quantization_config,
            temperature = kwargs.get('temperature', 0.0),
        )
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
        
    def __call__(self, prompt: str):
        input_ids = self.tokenizer(prompt, return_tensors="pt").to('cuda')
        output = self.model.generate(**input_ids, max_new_tokens=250)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)