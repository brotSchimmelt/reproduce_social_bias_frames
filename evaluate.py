import logging
import os
from datetime import datetime
from typing import Dict, List

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
    true_labels: List[str], predictions: List[str]
) -> Dict[str, float]:
    """Calculate F1 score, precision and recall for the positive classes."""
    positive_label = 1

    precision = precision_score(
        true_labels,
        predictions,
        pos_label=positive_label,
        average="binary",
        zero_division=0,
    )
    recall = recall_score(
        true_labels,
        predictions,
        pos_label=positive_label,
        average="binary",
        zero_division=0,
    )
    f1 = f1_score(
        true_labels,
        predictions,
        average="binary",
        pos_label=positive_label,
        zero_division=0,
    )

    return {"f1_score": f1, "precision": precision, "recall": recall}


def evaluate_generated_text(
    generated_text: str, reference_texts: List[str]
) -> Dict[str, float]:
    # nlp = spacy.load("en_core_web_md")

    # calculate BLEU-2
    reference_tokens = [ref.split() for ref in reference_texts]
    generated_tokens = generated_text.split()
    bleu_2_score = sentence_bleu(reference_tokens, generated_tokens, weights=(0.5, 0.5))

    # calculate RougeL
    rouge = Rouge()
    scores = [
        rouge.get_scores(generated_text, ref, avg=True) for ref in reference_texts
    ]
    rouge_l_f1 = np.mean([score["rouge-l"]["f"] for score in scores])

    # # calculate WMD
    # generated_doc = nlp(generated_text)
    # reference_docs = [nlp(ref) for ref in reference_texts]
    # wmd_scores = [generated_doc.similarity(ref_doc) for ref_doc in reference_docs]
    # wmd_score = np.mean(wmd_scores)

    # return {"bleu": bleu_2_score, "rouge": rouge_l_f1, "wmd": wmd_score}
    return {"bleu": bleu_2_score, "rouge": rouge_l_f1}
