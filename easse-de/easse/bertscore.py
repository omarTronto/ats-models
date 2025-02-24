from typing import List

from bert_score import BERTScorer

import easse.utils.preprocessing as utils_prep
from easse.utils.constants import LANGUAGE


def get_bertscore_sentence_scores(
    sys_sents: List[str],
    refs_sents: List[List[str]],
    lowercase: bool = False,
    tokenizer: str = "13a",
    language: str = LANGUAGE,
    tokenizer_obj=None,
):
    scorer = BERTScorer(lang=language, rescale_with_baseline=True)

    sys_sents = [utils_prep.normalize(sent, lowercase, tokenizer, tokenizer_obj=tokenizer_obj) for sent in sys_sents]
    refs_sents = [[utils_prep.normalize(sent, lowercase, tokenizer, tokenizer_obj=tokenizer_obj) for sent in ref_sents] for ref_sents in refs_sents]
    refs_sents = [list(r) for r in zip(*refs_sents)]

    return scorer.score(sys_sents, refs_sents)


def corpus_bertscore(
    sys_sents: List[str],
    refs_sents: List[List[str]],
    lowercase: bool = False,
    tokenizer: str = "13a",
    language: str = LANGUAGE,
    tokenizer_obj=None,
):
    all_scores = get_bertscore_sentence_scores(sys_sents, refs_sents, lowercase, tokenizer, language, tokenizer_obj)
    avg_scores = [s.mean(dim=0) for s in all_scores]
    precision = avg_scores[0].cpu().item()
    recall = avg_scores[1].cpu().item()
    f1 = avg_scores[2].cpu().item()
    return precision, recall, f1
