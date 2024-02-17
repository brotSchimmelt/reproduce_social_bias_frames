import logging

import pandas as pd
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer

import config
from utils.helper import clean_post


class SBICDataset(Dataset):
    def __init__(self, path: str, tokenizer: GPT2Tokenizer) -> None:
        self.tokenizer = tokenizer
        self.path = path
        self.df = pd.read_csv(self.path)
        self.generation_prompts, self.targets = [], []
        self.pad_value = tokenizer.pad_token_id
        self.max_length = config.MAX_LENGTH

        # iterate over the dataframe and prepare the data
        for _, row in self.df.iterrows():
            # preprocess input data
            post = clean_post(row["post"])
            lewd = 1 if row["sexYN"] == 1.0 else 0
            off = 1 if row["offensiveYN"] == 1.0 else 0
            intention = 1 if row["intentYN"] == 1.0 else 0
            grp = 1 if row["targetMinority"] else 0
            ing = 1 if row["speakerMinorityYN"] == 1.0 else 0
            group = row["targetMinority"]
            statement = row["targetStereotype"]

            # create samples
            generation_prompt = config.GENERATION_TEMPLATE.format(post=post).strip()

            # format target based on possible variable assignments
            if off == 0 and lewd == 0:
                target = config.OFFN_TARGET_TEMPLATE.format(
                    lewd=config.LEWD_TOKEN[lewd],
                    off=config.OFF_TOKEN[off],
                ).strip()
            elif off == 1 and grp == 0:
                target = config.OFFY_GRPN_TARGET_TEMPLATE.format(
                    lewd=config.LEWD_TOKEN[lewd],
                    off=config.OFF_TOKEN[off],
                    intention=config.INT_TOKEN[intention],
                    grp=config.GRP_TOKEN[grp],
                ).strip()
            else:
                target = config.FULL_TARGET_TEMPLATE.format(
                    lewd=config.LEWD_TOKEN[lewd],
                    off=config.OFF_TOKEN[off],
                    intention=config.INT_TOKEN[intention],
                    grp=config.GRP_TOKEN[grp],
                    group=group,
                    statement=statement,
                    ing=config.ING_TOKEN[ing],
                ).strip()

            self.generation_prompts.append(generation_prompt)
            self.targets.append(target)

        # remove filler tokens
        if not config.USE_FILL_TOKEN:
            self.generation_prompts = [
                p.replace(config.FILL_TOKEN, "") for p in self.generation_prompts
            ]
            self.targets = [t.replace(config.FILL_TOKEN, "") for t in self.targets]

        # tokenize the data
        self.encoded_prompts = tokenizer(
            self.generation_prompts,
            truncation=True,
            padding="max_length",  # NOTE change from True to 'max_length'
            max_length=config.MAX_LENGTH,
            return_tensors="pt",
        )
        self.encoded_targets = tokenizer(
            self.targets,
            truncation=True,
            padding="max_length",  # NOTE change from True to 'max_length'
            max_length=config.MAX_LENGTH,
            return_tensors="pt",
        )
        self.input_ids = self.encoded_prompts["input_ids"]
        self.attention_mask = self.encoded_prompts["attention_mask"]
        self.labels = self.encoded_targets["input_ids"]

        logging.info(
            f"len(Input_ids): {len(self.input_ids)} | len(labels): {len(self.labels)}"
        )
        logging.info(f"Input_id lengths: {set([len(i) for i in self.input_ids])}")
        logging.info(f"Label lengths: {set([len(i) for i in self.labels])}")
        logging.info(f"Use FILL token: {config.USE_FILL_TOKEN}")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }
