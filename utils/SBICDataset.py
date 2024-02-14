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
        self.generation_prompts = []
        self.targets = []

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
            target = config.TARGET_TEMPLATE.format(
                lewd=config.LEWD_TOKEN[lewd],
                off=config.OFF_TOKEN[off],
                intention=config.INT_TOKEN[intention],
                grp=config.GRP_TOKEN[grp],
                group=group,
                statement=statement,
                ing=config.ING_TOKEN[ing],
            ).strip()
            generation_prompt = config.INPUT_TEMPLATE.format(post=post).strip()

            self.generation_prompts.append(generation_prompt)
            self.targets.append(target)

        # tokenize the data
        self.encoded_prompts = tokenizer(
            self.generation_prompts,
            truncation=True,
            padding=True,
            max_length=config.MAX_LENGTH,
            return_tensors="pt",
        )
        self.encoded_targets = tokenizer(
            self.targets,
            truncation=True,
            padding=True,
            max_length=config.MAX_LENGTH,
            return_tensors="pt",
        )
        self.input_ids = self.encoded_prompts["input_ids"]
        self.attention_mask = self.encoded_prompts["attention_mask"]
        self.labels = self.encoded_targets["input_ids"]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }
