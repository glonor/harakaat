# This code is adapted from the following GitHub repository:
# https://github.com/Hamza5/Pipeline-diacritizer

# Modifications:
# - [List any significant modifications you made, if applicable]


import re
import sys

import numpy as np
from tqdm import tqdm

from src.utils.prepare_utils import strip_diacritics

D_NAMES = ["Fathatan", "Dammatan", "Kasratan", "Fatha", "Damma", "Kasra", "Shadda", "Sukun"]
NAME2DIACRITIC = {name: chr(code) for name, code in zip(D_NAMES, range(0x064B, 0x0653))}
ARABIC_DIACRITICS = frozenset(NAME2DIACRITIC.values())
ARABIC_LETTERS = frozenset([chr(x) for x in (list(range(0x0621, 0x63B)) + list(range(0x0641, 0x064B)))])
WORD_TOKENIZATION_REGEXP = re.compile(
    "((?:[" + "".join(ARABIC_LETTERS) + "][" + "".join(ARABIC_DIACRITICS) + r"]*)+|\d+(?:\.\d+)?)"
)
NUMBER_REGEXP = re.compile(r"\d+(?:\.\d+)?")


def extract_diacritics_2(text):
    """Return the diacritics from the text while keeping their original positions including the
    Shadda marks.

    :param text: str, the diacritized text.
    :return: list, the diacritics. Positions with double diacritics have a tuple as elements.
    """
    assert isinstance(text, str)
    diacritics = []
    for i in range(1, len(text)):
        if text[i] in ARABIC_DIACRITICS:
            if text[i - 1] == NAME2DIACRITIC["Shadda"]:
                diacritics[-1] = (text[i - 1], text[i])
            else:
                diacritics.append(text[i])
        elif text[i - 1] not in ARABIC_DIACRITICS:
            diacritics.append("")
    if text[-1] not in ARABIC_DIACRITICS:
        diacritics.append("")
    return diacritics


def der_wer_values(target_sentences, predicted_sentences, limit_to_arabic=True, include_no_diacritic=True):
    """Calculate the Diacritic Error Rate (DER), Word Error Rate (WER) values and Sentence
    Mismatch.

    :param target_sentences: list, the target sentences.
    :param predicted_sentences: list, the predicted sentences.
    :param limit_to_arabic: bool, whether to limit the calculation to Arabic characters.
    :param include_no_diacritic: bool, whether to include examples with no diacritic in the
        evaluation.
    :return: tuple, the DER, WER, DMER, WMER, and sentence mismatch values.
    """
    correct_d, correct_w, total_d, total_w, correct_dm, correct_wm, total_dm = 0, 0, 0, 0, 0, 0, 0
    sentences_mismatch = 0
    print("Calculating DER and WER values on {} characters".format("Arabic" if limit_to_arabic else "all"))
    print("{} no-diacritic Arabic letters".format("Including" if include_no_diacritic else "Ignoring"))
    for target_sentence, predicted_sentence in tqdm(zip(target_sentences, predicted_sentences)):
        if strip_diacritics(target_sentence) != strip_diacritics(predicted_sentence):
            sentences_mismatch += 1
            continue
        for orig_word, pred_word in zip(
            WORD_TOKENIZATION_REGEXP.split(target_sentence),
            WORD_TOKENIZATION_REGEXP.split(predicted_sentence),
        ):
            orig_word, pred_word = orig_word.strip(), pred_word.strip()
            if len(orig_word) == 0 or len(pred_word) == 0:  # Rare problematic scenario
                continue
            if limit_to_arabic:
                if not WORD_TOKENIZATION_REGEXP.match(orig_word) or NUMBER_REGEXP.match(orig_word):
                    continue
            orig_diacs = np.array([x[::-1] if len(x) == 2 else (x, "") for x in extract_diacritics_2(orig_word)])
            pred_diacs = np.array([x[::-1] if len(x) == 2 else (x, "") for x in extract_diacritics_2(pred_word)])
            if orig_diacs.shape != pred_diacs.shape:  # Rare problematic scenario
                print(
                    f"Diacritization mismatch between target and predicted forms: {orig_word} | {pred_word}",
                    file=sys.stderr,
                )
                continue
            if (
                not include_no_diacritic
                and WORD_TOKENIZATION_REGEXP.match(orig_word)
                and not NUMBER_REGEXP.match(orig_word)
            ):
                diacritics_indexes = orig_diacs[:, 0] != ""
                pred_diacs = pred_diacs[diacritics_indexes]
                orig_diacs = orig_diacs[diacritics_indexes]
            correct_w += np.all(orig_diacs == pred_diacs)
            correct_wm += np.all(orig_diacs[:-1] == pred_diacs[:-1])
            total_w += 1
            correct_d += np.sum(np.all(orig_diacs == pred_diacs, axis=1))
            correct_dm += np.sum(np.all(orig_diacs[:-1] == pred_diacs[:-1], axis=1))
            total_d += orig_diacs.shape[0]
            total_dm += orig_diacs[:-1].shape[0]

    score_d = 1 - (correct_d / total_d) if total_d != 0 else 0
    score_w = 1 - (correct_w / total_w) if total_w != 0 else 0
    score_dm = 1 - (correct_dm / total_dm) if total_dm != 0 else 0
    score_w = 1 - (correct_wm / total_w) if total_w != 0 else 0
    score_sentences = sentences_mismatch / len(target_sentences) if len(target_sentences) != 0 else 0

    return (score_d, score_w, score_dm, score_w, score_sentences)
