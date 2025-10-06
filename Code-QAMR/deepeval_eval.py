import transformers
from constants.config import *
from constants.helpers import *
from deepeval.models import DeepEvalBaseLLM
from pydantic import BaseModel
import json
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import (
    build_transformers_prefix_allowed_tokens_fn,
)

class CustomLlama3_70B(DeepEvalBaseLLM):
    def __init__(self):
        quantization_config = bnb_config

        model_4bit = get_model(model_path,model_id)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_path,
            token=hf_auth
        )

        self.model = model_4bit
        self.tokenizer = tokenizer

    def load_model(self):
        return self.model
    def generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        # Same as the previous example above
        model = self.load_model()
        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            use_cache=True,
            device_map="auto",
            max_new_tokens=5000,
            do_sample=True,
            top_k=5,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # Create parser required for JSON confinement using lmformatenforcer
        parser = JsonSchemaParser(schema.schema())
        prefix_function = build_transformers_prefix_allowed_tokens_fn(
            pipeline.tokenizer, parser
        )
        import re
        # Output and load valid JSON
        output_dict = pipeline(prompt, prefix_allowed_tokens_fn=prefix_function)
        output = output_dict[0]["generated_text"][len(prompt) :]
        with open("deepeval_steps.txt", "a") as file:
            file.write(f"{output}\n-------\n")
        output = output.replace('\\"', "'")
        output = re.sub(r'("reason":\s*")([^"]*?)"', lambda m: m.group(1) + m.group(2).replace('"', "'") + '"', output)
        if output.count('"') % 2 != 0:
            # Add a closing quote at the end if needed
            output += '"'
        # Add a closing brace if missing
        if output.count('{') > output.count('}'):
            output += '}'

        # output = re.sub(r'\\\'', "'", output)
        # output = re.sub(r'(\}(\s*\{|\s*\]))', r'},\2', output)
        # output = re.sub(r',(\s*[\}\]])', r'\1', output)


        with open("deepeval_steps.txt", "a") as file:
            file.write(f"{output}\n-------\n")
            try:
            # Fix the JSON string before loading
                json_result = json.loads(output,strict=False)
            except json.JSONDecodeError as e:
                file.write(f"Error decoding JSON: {e}\n\n")
                return

        # Return valid JSON object according to the schema DeepEval supplied
        return schema(**json_result)

    async def a_generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        return self.generate(prompt, schema)

    def get_model_name(self):
        return "Llama-3 70B"