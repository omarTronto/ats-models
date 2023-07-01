#! usr/bin/env bash

python <<HEREDOC
from scripts import finetune_trimmed_mbart, finetune_trimmed_longmbart

sents_datasets = [
    'datasets/prepared_datasets/sents/DEplain-APA',
    'datasets/prepared_datasets/sents/DEplain-APA-web'
]

docs_datasets = [
    'datasets/prepared_datasets/docs/DEplain-APA',
    'datasets/prepared_datasets/docs/DEplain-web',
    'datasets/prepared_datasets/docs/DEplain-APA-web'
]

finetune_trimmed_mbart(
    datasets=sents_datasets,
    vocab_size_in_k='35',
    trial='trial_1',
    max_input_len=256,
    max_output_len=256,
    attention_window=512,
    batch_size=16,
    grad_accum=1,
    num_workers=20,
    seed=1234
)

finetune_trimmed_longmbart(
    datasets=docs_datasets,
    vocab_size_in_k='35',
    trial='trial_1',
    max_input_len=2048,
    max_output_len=1024,
    attention_window=512,
    batch_size=1,
    grad_accum=1,
    num_workers=20,
    seed=1234
)

HEREDOC