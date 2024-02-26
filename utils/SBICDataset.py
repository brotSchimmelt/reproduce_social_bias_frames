import logging
from typing import Dict

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer

import config
from utils.helper import clean_post


class SBICDataset(Dataset):
    def __init__(self, path: str, tokenizer: GPT2Tokenizer) -> None:
        self.tokenizer = tokenizer
        self.path = path
        self.df = pd.read_csv(self.path)
        self.texts = []

        # iterate over the dataframe and prepare the data
        for _, row in self.df.iterrows():
            # preprocess input data
            post = clean_post(row["post"])
            lewd = 1 if row["sexYN"] == 1.0 else 0
            off = 1 if row["offensiveYN"] == 1.0 else 0
            intention = 1 if row["intentYN"] == 1.0 else 0
            grp = 1 if row["whoTarget"] == 1.0 else 0
            ing = 1 if row["speakerMinorityYN"] == 1.0 else 0
            group = (
                row["targetMinority"] if isinstance(row["targetMinority"], str) else ""
            )
            statement = (
                row["targetStereotype"]
                if isinstance(row["targetStereotype"], str)
                else ""
            )

            # create samples based on possible variable assignments
            if off == 0 and lewd == 0:
                self.texts.append(
                    config.TRAIN_TEMPLATE_OFFN.format(
                        post=post,
                        lewd=config.LEWD_TOKEN[lewd],
                        off=config.OFF_TOKEN[off],
                    ).strip()
                )

            elif off == 1 and grp == 0:
                self.texts.append(
                    config.TRAIN_TEMPLATE_GRPN.format(
                        post=post,
                        lewd=config.LEWD_TOKEN[lewd],
                        off=config.OFF_TOKEN[off],
                        intention=config.INT_TOKEN[intention],
                        grp=config.GRP_TOKEN[grp],
                    ).strip()
                )

            else:
                self.texts.append(
                    config.TRAIN_TEMPLATE_FULL.format(
                        post=post,
                        lewd=config.LEWD_TOKEN[lewd],
                        off=config.OFF_TOKEN[off],
                        intention=config.INT_TOKEN[intention],
                        grp=config.GRP_TOKEN[grp],
                        group=group,
                        statement=statement,
                        ing=config.ING_TOKEN[ing],
                    ).strip()
                )

        # tokenize the data
        self.encoding_dict = tokenizer(
            self.texts,
            truncation=True,
            max_length=config.MAX_LENGTH,
            padding="max_length",
            return_tensors="pt",
        )
        self.input_ids = self.encoding_dict["input_ids"]
        self.attention_mask = self.encoding_dict["attention_mask"]
        logging.info(f"First text: {self.texts[0]}")

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        input_ids = self.input_ids[idx]
        attention_mask = self.attention_mask[idx]
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
