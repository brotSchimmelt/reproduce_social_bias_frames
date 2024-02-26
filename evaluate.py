import logging
import os
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import spacy
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from sklearn.metrics import f1_score, precision_score, recall_score

import config

#### LOGGING ####
if not os.path.exists(config.LOG_DIR):
    os.makedirs(config.LOG_DIR)
app_name = os.path.splitext(os.path.basename(__file__))[0]
start = datetime.now()
timestamp = start.strftime(config.LOGGING_FILE_DATEFMT)
logging.basicConfig(
    filename=f"{config.LOG_DIR}{app_name}_{timestamp}.log",
    level=config.LOGGING_LEVEL,
    format=config.LOGGING_FORMAT,
    datefmt=config.LOGGING_DATEFMT,
)
#### LOGGING ####


def evaluate_categorical_variables(
    predictions: List[str], targets: List[str]
) -> Dict[str, float]:
    """Calculate F1 score, precision and recall for the positive classes."""
    positive_label = "positive"

    # Calculate metrics
    precision = precision_score(
        targets, predictions, pos_label=positive_label, zero_division=0
    )
    recall = recall_score(
        targets, predictions, pos_label=positive_label, zero_division=0
    )
    f1 = f1_score(targets, predictions, pos_label=positive_label, zero_division=0)

    return {"precision": precision, "recall": recall, "f1_score": f1}


def evaluate_generated_text(
    generated_text: List[str], reference_texts: List[str]
) -> Dict[str, float]:
    # Load SpaCy model for WMD
    nlp = spacy.load("en_core_web_md")

    # Calculate BLEU-2
    reference_tokens = [ref.split() for ref in reference_texts]
    generated_tokens = generated_text.split()
    bleu_2_score = sentence_bleu(reference_tokens, generated_tokens, weights=(0.5, 0.5))

    # Calculate RougeL
    rouge = Rouge()
    scores = rouge.get_scores(generated_text, reference_texts, avg=True)
    rouge_l_f1 = scores["rouge-l"]["f"]

    # Calculate WMD
    generated_doc = nlp(generated_text)
    reference_docs = [nlp(ref) for ref in reference_texts]
    wmd_scores = [generated_doc.similarity(ref_doc) for ref_doc in reference_docs]
    wmd_score = np.mean(wmd_scores)

    return {"BLEU-2": bleu_2_score, "RougeL F1": rouge_l_f1, "WMD": wmd_score}


def split_output(model_output: str) -> Dict[str, Any]:
    """Split the model output into the different parts."""

    # get the categorical variables
    predictions = {
        "lewd": 1 if config.LEWD_TOKEN[1] in model_output else 0,
        "off": 1 if config.OFF_TOKEN[1] in model_output else 0,
        "intention": 1 if config.INT_TOKEN[1] in model_output else 0,
        "grp": 1 if config.GRP_TOKEN[1] in model_output else 0,
        "ing": 1 if config.ING_TOKEN[1] in model_output else 0,
    }

    # TODO find text parts

    return predictions
