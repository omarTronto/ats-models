from typing import List

import numpy as np
import textstat

import easse.utils.preprocessing as utils_prep
from easse.utils.constants import LANGUAGE


def corpus_fre(
    sys_sents: List[str],
    lowercase: bool = False,
    tokenizer: str = "13a",
    language: str = LANGUAGE,
    tokenizer_obj=None,
):
    textstat.textstat.set_lang(language)
    sys_sents = [utils_prep.normalize(sent, lowercase, tokenizer, tokenizer_obj=tokenizer_obj) for sent in sys_sents]
    fre_score = textstat.textstat.flesch_reading_ease(" ".join(sys_sents))
    return fre_score


def sentence_fre(
    sys_sent: str,
    lowercase: bool = False,
    tokenizer: str = "13a",
    language: str = LANGUAGE,
    tokenizer_obj=None,
):
    return corpus_fre(
        [sys_sent],
        lowercase=lowercase,
        tokenizer=tokenizer,
        tokenizer_obj=tokenizer_obj,
        language=language,
    )


def corpus_averaged_sentence_fre(
    sys_sents: List[str],
    lowercase: bool = False,
    tokenizer: str = "13a",
    tokenizer_obj=None,
    language: str = LANGUAGE,
):

    scores = []
    for sys_sent in sys_sents:
        scores.append(
            sentence_fre(
                sys_sent,
                lowercase=lowercase,
                tokenizer=tokenizer,
                tokenizer_obj=tokenizer_obj,
            )
        )
    return np.mean(scores)


def corpus_wiener(
    sys_sents: List[str],
    lowercase: bool = False,
    tokenizer: str = "13a",
    language: str = LANGUAGE,
    number: int = 1,
    tokenizer_obj=None,
):
    textstat.textstat.set_lang(language)
    sys_sents = [utils_prep.normalize(sent, lowercase, tokenizer, tokenizer_obj=tokenizer_obj) for sent in sys_sents]
    try:
        wiener_score = textstat.textstat.wiener_sachtextformel(" ".join(sys_sents), number)
    except ZeroDivisionError:
        wiener_score = np.nan
    return wiener_score


def sentence_wiener(
    sys_sent: str,
    lowercase: bool = False,
    tokenizer: str = "13a",
    language: str = LANGUAGE,
    number: int = 1,
    tokenizer_obj=None,
):
    return corpus_wiener([sys_sent], number=number,
                         lowercase=lowercase, tokenizer=tokenizer, tokenizer_obj=tokenizer_obj, language=language,
                         )


def corpus_averaged_sentence_wiener(
    sys_sents: List[str],
    lowercase: bool = False,
    tokenizer: str = "13a",
    language: str = LANGUAGE,
    number: int = 1,
    tokenizer_obj=None,
):
    textstat.textstat.set_lang(language)
    scores = list()
    for sent in sys_sents:
        sent = utils_prep.normalize(sent, lowercase, tokenizer, tokenizer_obj=tokenizer_obj)
        scores.append(sentence_wiener(sent, number=number, lowercase=lowercase, tokenizer=tokenizer, tokenizer_obj=tokenizer_obj, language=language,))
    return round(np.nanmean(scores),4)


def corpus_wiener_1(
    sys_sents: List[str],
    lowercase: bool = False,
    tokenizer: str = "13a",
    language: str = LANGUAGE,
    tokenizer_obj=None,
):
    return corpus_wiener(sys_sents, lowercase, tokenizer, language, 1, tokenizer_obj=tokenizer_obj)


def corpus_wiener_2(
    sys_sents: List[str],
    lowercase: bool = False,
    tokenizer: str = "13a",
    language: str = LANGUAGE,
    tokenizer_obj=None,
):
    return corpus_wiener(sys_sents, lowercase, tokenizer, language, 2, tokenizer_obj=tokenizer_obj)


def corpus_wiener_3(
    sys_sents: List[str],
    lowercase: bool = False,
    tokenizer: str = "13a",
    language: str = LANGUAGE,
    tokenizer_obj=None,
):
    return corpus_wiener(sys_sents, lowercase, tokenizer, language, 3, tokenizer_obj=tokenizer_obj)


def corpus_wiener_4(
    sys_sents: List[str],
    lowercase: bool = False,
    tokenizer: str = "13a",
    language: str = LANGUAGE,
    tokenizer_obj=None,
):
    return corpus_wiener(sys_sents, lowercase, tokenizer, language, 4, tokenizer_obj=tokenizer_obj)


def sent_wiener_1(
    sys_sents: List[str],
    lowercase: bool = False,
    tokenizer: str = "13a",
    language: str = LANGUAGE,
    tokenizer_obj=None,
):
    return corpus_averaged_sentence_wiener(sys_sents, lowercase, tokenizer, language, 1, tokenizer_obj=tokenizer_obj)


def sent_wiener_2(
    sys_sents: List[str],
    lowercase: bool = False,
    tokenizer: str = "13a",
    language: str = LANGUAGE,
    tokenizer_obj=None,
):
    return corpus_averaged_sentence_wiener(sys_sents, lowercase, tokenizer, language, 2, tokenizer_obj=tokenizer_obj)


def sent_wiener_3(
    sys_sents: List[str],
    lowercase: bool = False,
    tokenizer: str = "13a",
    language: str = LANGUAGE,
    tokenizer_obj=None,
):
    return corpus_averaged_sentence_wiener(sys_sents, lowercase, tokenizer, language, 3, tokenizer_obj=tokenizer_obj)


def sent_wiener_4(
    sys_sents: List[str],
    lowercase: bool = False,
    tokenizer: str = "13a",
    language: str = LANGUAGE,
    tokenizer_obj=None,
):
    return corpus_averaged_sentence_wiener(sys_sents, lowercase, tokenizer, language, 4, tokenizer_obj=tokenizer_obj)
