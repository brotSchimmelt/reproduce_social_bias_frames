from typing import Dict, List

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

import config
from utils.helper import get_device


class CustomGenerator:
    def __init__(
        self,
        model: GPT2LMHeadModel,
        tokenizer: GPT2Tokenizer,
        decoding_strategy: str,
        device: str = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.strategy = decoding_strategy
        self.device = device if device else get_device()

        if self.strategy not in ["greedy", "sample", "constrained", "all"]:
            raise ValueError(f"Invalid decoding strategy: {self.strategy}")

        self.generator = pipeline(
            "text-generation", model=model, tokenizer=tokenizer, device=0
        )

    def generate(self, generation_prompt: str) -> Dict[str, List[str]]:
        if self.strategy == "greedy":
            return {
                "generated_text": self.decode_greedy(generation_prompt),
            }

        elif self.strategy == "sample":
            return {
                "generated_text": self.decode_sampling(generation_prompt),
            }

        elif self.strategy == "constrained":
            return {
                "generated_text": self.decode_constraint(generation_prompt),
            }

        else:
            return {
                "greedy": self.decode_greedy(generation_prompt),
                "sample": self.decode_sampling(generation_prompt),
                "constrained": self.decode_constraint(generation_prompt),
            }

    def decode_greedy(self, generation_prompt: str) -> List[str]:
        """Greedy decoding."""
        output = self.generator(
            generation_prompt,
            max_length=config.MAX_LENGTH,
            num_return_sequences=1,
            do_sample=False,
        )
        return [output[0]["generated_text"]]

    def decode_sampling(self, generation_prompt: str) -> List[str]:
        """Sample decoding."""
        output = self.generator(
            generation_prompt,
            max_length=config.MAX_LENGTH,
            num_return_sequences=config.DEFAULT_NUM_RETURN_SEQ,
            do_sample=True,
        )
        return [item["generated_text"] for item in output]

    def decode_constraint(self, generation_prompt: str) -> List[str]:
        """Constrained decoding."""
        self.model.eval()
        generated_ids = self.tokenizer.encode(
            generation_prompt, add_special_tokens=True, return_tensors="pt"
        ).to(self.device)

        for _ in range(config.MAX_LENGTH):
            # generate next token
            with torch.no_grad():
                outputs = self.model(generated_ids)
            logits = outputs.logits
            next_token_logits = logits[0, -1, :]
            next_token_id = next_token_logits.argmax().unsqueeze(0).unsqueeze(0)

            # TODO add logic here

            # add next token to sequence of generated tokens
            generated_ids = torch.cat((generated_ids, next_token_id), dim=1)

            if next_token_id.item() == self.tokenizer.encode(config.END_TOKEN)[0]:
                break

        return [
            self.tokenizer.decode(generated_ids[0].cpu(), skip_special_tokens=False)
        ]

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return f"CustomDecoder(strategy={self.strategy})"
