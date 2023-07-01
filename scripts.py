import os
import subprocess
import ast
import pandas as pd
import numpy as np
import shutil

def prepare_trimmed_mbart_model(trimmed_vocab_size_in_k):
    
    base_model = 'facebook/mbart-large-cc25'
    
    available_vocab_sizes = ['all', '20', '23', '25', '27', '30', '35']
    assert trimmed_vocab_size_in_k in available_vocab_sizes, f"Vocab size {trimmed_vocab_size_in_k} is not available. Please choose from {available_vocab_sizes}"
    
    if trimmed_vocab_size_in_k == 'all':
        trimmed_vocab_size_in_k = 'filtered'
    
    model_outpt_path = f'pretrained_models/trimmed_mbart_{trimmed_vocab_size_in_k}k' if trimmed_vocab_size_in_k != 'filtered' else f'pretrained_models/trimmed_mbart_{trimmed_vocab_size_in_k}'
    base_model_cache = f'pretrained_models/{base_model}'
    vocab_path = f'vocab_lists/mbart/all.spm.uniq.{trimmed_vocab_size_in_k}k' if trimmed_vocab_size_in_k != 'filtered' else f'vocab_lists/mbart/all.spm.uniq.{trimmed_vocab_size_in_k}' 
    
    # prepare the folder structure and copy the missing sentencepiece model
    if not os.path.exists(model_outpt_path):
        os.makedirs(model_outpt_path)
    if not os.path.exists(base_model_cache):
        os.makedirs(base_model_cache)
    if not os.path.exists(f'{base_model_cache}/sentencepiece.bpe.model'):
        shutil.copy("missing_sentencepiece_model/sentencepiece.bpe.model", base_model_cache)
    
    output = subprocess.check_output(f"""python -m ats_models.trim_mbart \
                                        --base_model {base_model} \
                                        --tokenizer {base_model} \
                                        --save_model_to {model_outpt_path} \
                                        --cache_dir {base_model_cache} \
                                        --reduce_to_vocab {vocab_path} \
                                        --add_language_tags de_SI \
                                        --initialize_tags de_DE """, shell=True, text=True)
    
    return 

def prepare_trimmed_longmbart_model(trimmed_vocab_size_in_k, max_pos=2048, attention_window=512):
    
    base_model = 'facebook/mbart-large-cc25'
    
    available_vocab_sizes = ['all', '20', '23', '25', '27', '30', '35']
    assert trimmed_vocab_size_in_k in available_vocab_sizes, f"Vocab size {trimmed_vocab_size_in_k} is not available. Please choose from {available_vocab_sizes}"
    
    if trimmed_vocab_size_in_k == 'all':
        trimmed_vocab_size_in_k = 'filtered'
    
    model_outpt_path = f'pretrained_models/trimmed_longmbart_{trimmed_vocab_size_in_k}k' if trimmed_vocab_size_in_k != 'filtered' else f'pretrained_models/trimmed_longmbart_{trimmed_vocab_size_in_k}'
    base_model_cache = f'pretrained_models/{base_model}'
    vocab_path = f'vocab_lists/mbart/all.spm.uniq.{trimmed_vocab_size_in_k}k' if trimmed_vocab_size_in_k != 'filtered' else f'vocab_lists/mbart/all.spm.uniq.{trimmed_vocab_size_in_k}' 
    
    # prepare the folder structure and copy the missing sentencepiece model
    if not os.path.exists(model_outpt_path):
        os.makedirs(model_outpt_path)
    if not os.path.exists(base_model_cache):
        os.makedirs(base_model_cache)
    if not os.path.exists(f'{base_model_cache}/sentencepiece.bpe.model'):
        shutil.copy("missing_sentencepiece_model/sentencepiece.bpe.model", base_model_cache)
    
    output = subprocess.check_output(f"""python -m ats_models.convert_mbart2long \
                                        --base_model {base_model} \
                                        --tokenizer {base_model} \
                                        --save_model_to {model_outpt_path} \
                                        --cache_dir {base_model_cache} \
                                        --max_pos {max_pos} \
                                        --attention_window {attention_window} \
                                        --reduce_to_vocab {vocab_path} \
                                        --add_language_tags de_SI \
                                        --initialize_tags de_DE """, shell=True, text=True)
    
    return

def finetune_trimmed_mbart(
    datasets, 
    vocab_size_in_k, 
    trial, 
    max_output_len=256, 
    max_input_len=256, 
    attention_window=512, 
    batch_size=16, 
    grad_accum=1,
    num_workers=20,
    seed=1,
):
    if os.path.exists(f'finetuned_checkpoints/{trial}'):
        print(f"Finetuning for trial {trial} already exists. Skipping...")
        return
    
    for dataset in datasets:
        
        dataset_name = dataset.split('/')[-1]
        finetuned_dir = f'finetuned_checkpoints/{trial}/trimmed_mbart_{vocab_size_in_k}k_{dataset_name}'
        
        if not os.path.exists(finetuned_dir):
            os.makedirs(finetuned_dir)
        
        pretrained_dir = f'pretrained_models/trimmed_mbart_{vocab_size_in_k}k'
        
        output = subprocess.check_output(f"""python -m ats_models.finetune_mbart \
                                            --from_pretrained {pretrained_dir} \
                                            --tokenizer {pretrained_dir} \
                                            --save_dir {'/'.join(finetuned_dir.split('/')[:2])} \
                                            --save_prefix {finetuned_dir.split('/')[-1]} \
                                            --train_source {dataset}/train.src \
                                            --train_target {dataset}/train.tgt \
                                            --dev_source {dataset}/dev.src \
                                            --dev_target {dataset}/dev.tgt \
                                            --test_source {dataset}/test.src \
                                            --test_target {dataset}/test.tgt \
                                            --max_output_len {max_output_len} \
                                            --max_input_len {max_input_len} \
                                            --attention_mode n2 \
                                            --attention_window {attention_window} \
                                            --batch_size {batch_size} \
                                            --grad_accum {grad_accum} \
                                            --num_workers {num_workers} \
                                            --accelerator gpu \
                                            --devices 0 \
                                            --seed {seed} \
                                            --attention_dropout 0.1 \
                                            --dropout 0.3 \
                                            --label_smoothing 0.2 \
                                            --lr 0.00003 \
                                            --check_val_every_n_epoch 1 \
                                            --val_percent_check 1.0 \
                                            --test_percent_check 1.0 \
                                            --early_stopping_metric 'rougeL' \
                                            --patience 10 \
                                            --min_delta 0.0005 \
                                            --lr_reduce_patience 8 \
                                            --lr_reduce_factor 0.5 \
                                            --grad_ckpt \
                                            --disable_validation_bar \
                                            --progress_bar_refresh_rate 10 \
                                            --save_top_k 3 \
                                            --src_lang de_DE \
                                            --tgt_lang de_SI >> {finetuned_dir}/training_logs """, shell=True, text=True)
        
    return 

def finetune_trimmed_longmbart(
    datasets, 
    vocab_size_in_k, 
    trial, 
    max_output_len=1024, 
    max_input_len=2048, 
    attention_window=512, 
    batch_size=1, 
    grad_accum=1,
    num_workers=20,
    seed=1,
):
    
    for dataset in datasets:
        
        dataset_name = dataset.split('/')[-1]
        finetuned_dir = f'finetuned_checkpoints/{trial}/trimmed_longmbart_{vocab_size_in_k}k_{dataset_name}'
        
        if not os.path.exists(finetuned_dir):
            os.makedirs(finetuned_dir)
        
        pretrained_dir = f'pretrained_models/trimmed_longmbart_{vocab_size_in_k}k'
        
        output = subprocess.check_output(f"""python -m ats_models.finetune_mbart \
                                            --from_pretrained {pretrained_dir} \
                                            --tokenizer {pretrained_dir} \
                                            --save_dir {'/'.join(finetuned_dir.split('/')[:2])} \
                                            --save_prefix {finetuned_dir.split('/')[-1]} \
                                            --train_source {dataset}/train.src \
                                            --train_target {dataset}/train.tgt \
                                            --dev_source {dataset}/dev.src \
                                            --dev_target {dataset}/dev.tgt \
                                            --test_source {dataset}/test.src \
                                            --test_target {dataset}/test.tgt \
                                            --max_output_len {max_output_len} \
                                            --max_input_len {max_input_len} \
                                            --attention_mode sliding_chunks \
                                            --attention_window {attention_window} \
                                            --batch_size {batch_size} \
                                            --grad_accum {grad_accum} \
                                            --num_workers {num_workers} \
                                            --accelerator gpu \
                                            --devices 0 \
                                            --seed {seed} \
                                            --attention_dropout 0.1 \
                                            --dropout 0.3 \
                                            --label_smoothing 0.2 \
                                            --lr 0.00003 \
                                            --check_val_every_n_epoch 1 \
                                            --val_percent_check 1.0 \
                                            --test_percent_check 1.0 \
                                            --early_stopping_metric 'rougeL' \
                                            --patience 10 \
                                            --min_delta 0.0005 \
                                            --lr_reduce_patience 8 \
                                            --lr_reduce_factor 0.5 \
                                            --grad_ckpt \
                                            --disable_validation_bar \
                                            --progress_bar_refresh_rate 10 \
                                            --save_top_k 3 \
                                            --src_lang de_DE \
                                            --tgt_lang de_SI >> {finetuned_dir}/training_logs """, shell=True, text=True)
        
    return 

def evaluate_baseline_src2src(docs_test_sets, sents_test_sets):
    df_rows = []
    
    if not os.path.exists('eval_metrics'):
        os.mkdir('eval_metrics')
    
    # evaluate document level test sets
    print("Evaluating document level test sets...")
    print("="*50)
    print("="*50)
    
    for test_set in docs_test_sets:
        src_file_path = f"{test_set}/test.src"
        tgt_file_path = f"{test_set}/test.tgt"
        out_path = src_file_path
           
        # evaluate the outputs using easse-de
        print(f"====>> Evaluating src2src model on {test_set.split('/')[-1]}.........")
        remove_lang_tags(out_path)
        output = subprocess.check_output(f"""easse evaluate \
                        -lang de \
                        -il document-level \
                        -m bleu,sari,bertscore,fre_sent,fre_corpus \
                        -t custom \
                        --orig_sents_path {src_file_path} \
                        --refs_sents_paths {tgt_file_path} \
                        -i {out_path} """, shell=True, text=True)
        
        metrics_str = output.split('\n')[-2]
        metrics_dict = ast.literal_eval(metrics_str)
                
        metrics_dict['model'] = 'src2src'
        
        metrics_dict['level'] = 'document'
        
        df_rows.append(metrics_dict)
        
        print("="*50)
        print("="*50)
        print(f"Finished Evaluating src2src on {test_set.split('/')[-1]} successfully.")
        print(f"Evaluation metrics: {metrics_dict}")
        print("="*50)
        print("="*50)
            
    print("="*50)
    print("="*50)
    
    print("Evaluating sentence level sets...")
    print("="*50)
    print("="*50)
    
    for test_set in sents_test_sets:
        src_file_path = f"{test_set}/test.src"
        tgt_file_path = f"{test_set}/test.tgt"
        out_path = src_file_path
        
        # evaluate the outputs using easse-de

        print(f"====>> Evaluating src2src model on {test_set.split('/')[-1]}.........")
        output = subprocess.check_output(f"""easse evaluate \
                        -lang de \
                        -il sentence-level \
                        -m bleu,sari,bertscore,fre_sent,fre_corpus \
                        -t custom \
                        --orig_sents_path {src_file_path} \
                        --refs_sents_paths {tgt_file_path} \
                        -i {out_path} """, shell=True, text=True)
        
        metrics_str = output.split('\n')[-2]
        metrics_dict = ast.literal_eval(metrics_str)
        
        metrics_dict['model'] = 'src2src'
        metrics_dict['test_set'] = test_set.split('/')[-1]
        metrics_dict['level'] = 'sentence'
        
        df_rows.append(metrics_dict)         
        
        print("="*50)
        print("="*50)
        print(f"Finished Evaluating src2src model on {test_set} successfully.")
        print("="*50)
        print("="*50)
    
    df = pd.DataFrame(df_rows)
    df = df[['model',
             'test_set', 
             'level',  
             'bleu',
             'sari', 
             'bertscore_precision', 
             'bertscore_recall', 
             'bertscore_f1', 
             'sent_FRE', 
             'corpus_FRE'
             ]]
    
    print("="*50)
    print("="*50)
    print("="*50)
    print("="*50)
    
    print('Saving evaluation metrics to csv file...')
    
    df.to_csv(f"eval_metrics/src2src.csv", index=False)
    
    print("="*50)
    print("="*50)
    print("="*50)
    print("="*50)
    
    print('Done!')
        
    return df

def evaluate_trial(trial_name, docs_test_sets, sents_test_sets, beam_size=6, generate_outputs=False, anlaysis=False):
    
    docs_models_paths, sents_models_paths = get_models_paths(trial_name)
    df_rows = []
    old_csv = None
    
    if not os.path.exists('eval_metrics'):
        os.mkdir('eval_metrics')
    
    if generate_outputs:
        if not os.path.exists(f'outputs/{trial_name}'):
            os.makedirs(f'outputs/{trial_name}')
        
        if os.path.exists(f'eval_metrics/{trial_name}.csv'):
            old_csv = pd.read_csv(f'eval_metrics/{trial_name}.csv')
    
    analys = ' -q -a ' if anlaysis else ' '
    
    # evaluate document level models
    print("Evaluating document level models")
    print("="*50)
    print("="*50)
    for model_path in docs_models_paths:
        print(f"Evaluating {model_path}")
        print("-"*50)
        
        # inference on test sets of APA, web, and 20Min.
        for test_set in docs_test_sets:
            if generate_outputs:
                print(f"====>> Generating outputs using {model_path.split('/')[2]} on {test_set.split('/')[-1]}.........")
            src_file_path = f"{test_set}/test.src"
            tgt_file_path = f"{test_set}/test.tgt"
            out_path = f"outputs/{trial_name}/{model_path.split('/')[2]}_on_{test_set.split('/')[-1]}"
            model_path_arg = '/'.join(model_path.split('/')[:-1])
            checkpoint_arg = model_path.split('/')[-1]
            rougeL = None
            if generate_outputs:
                output = subprocess.check_output(f"""python -m ats_models.inference_mbart \
                                        --model_path {model_path_arg} \
                                        --checkpoint {checkpoint_arg} \
                                        --tokenizer {model_path_arg} \
                                        --test_source {src_file_path} \
                                        --test_target {tgt_file_path} \
                                        --src_lang de_DE \
                                        --tgt_lang de_SI \
                                        --is_long \
                                        --max_input_len 2048 \
                                        --max_output_len 1024 \
                                        --batch_size 1 \
                                        --num_workers 20 \
                                        --accelerator gpu \
                                        --devices 0 \
                                        --beam_size {beam_size} \
                                        --progress_bar_refresh_rate 1 \
                                        --translation {out_path} """, shell=True, text=True)
            
                for line in output.split('\n')[-10:]:
                    if 'rougeL:' in line:
                        rougeL = float(line.split('[')[-1][:-2])
                
                assert rougeL is not None, f"Could not find rougeL in {output} for {model_path} on {test_set}"
            
                print("="*50)
                print(f"Finished generating outputs using {model_path.split('/')[2]} on {test_set.split('/')[-1]} successfully.")
                print("="*50)
            
            
            # evaluate the outputs using easse-de
            print(f"====>> Evaluating generated outputs of {model_path.split('/')[2]} on {test_set.split('/')[-1]}.........")
            remove_lang_tags(out_path)
            output = subprocess.check_output(f"""easse evaluate \
                            -lang de \
                            -il document-level \
                            -m bleu,sari,bertscore,fre_sent,fre_corpus \
                            -t custom \
                            --orig_sents_path {src_file_path} \
                            --refs_sents_paths {tgt_file_path} \
                            {analys} -i {out_path} """, shell=True, text=True)
            
            metrics_str = output.split('\n')[-2]
            metrics_dict = ast.literal_eval(metrics_str)
            if anlaysis:
                quality_estimation = metrics_dict['quality_estimation']
            
            metrics_dict['compression_ratio'] = quality_estimation['Compression ratio'] if anlaysis else " "
            metrics_dict['sentence_splits'] = quality_estimation['Sentence splits'] if anlaysis else " "
            metrics_dict['levenshtein_similarity'] = quality_estimation['Levenshtein similarity'] if anlaysis else " "
            metrics_dict['exact_copies'] = quality_estimation['Exact copies'] if anlaysis else " "
            metrics_dict['additions_proportion'] = quality_estimation['Additions proportion'] if anlaysis else " "
            metrics_dict['deletions_proportion'] = quality_estimation['Deletions proportion'] if anlaysis else " "
            metrics_dict['lexical_complexity_score'] = quality_estimation['Lexical complexity score'] if anlaysis else " "
            
            metrics_dict['model'] = ' '.join(model_path.split('/')[2].split('_')[:2])
            metrics_dict['train_set'] = ' '.join(model_path.split('/')[2].split('_')[3:])
            metrics_dict['test_set'] = test_set.split('/')[-1]
            if old_csv is not None:
                rougeL = old_csv[(old_csv['model'] == metrics_dict['model']) & (old_csv['train_set'] == metrics_dict['train_set']) & (old_csv['test_set'] == metrics_dict['test_set'])]['rougeL'].values[0]
            metrics_dict['rougeL'] = rougeL if rougeL is not None else np.nan
            metrics_dict['level'] = 'document'
            
            df_rows.append(metrics_dict)         
            
            print("="*50)
            print("="*50)
            print(f"Finished Evaluating {model_path.split('/')[2]} on {test_set.split('/')[-1]} successfully.")
            print(f"Evaluation metrics: {metrics_dict}")
            print("="*50)
            print("="*50)
            
    print("="*50)
    print("="*50)
    
    # evaluate sentence level models
    print("Evaluating sentence level models")
    print("="*50)
    print("="*50)
    for model_path in sents_models_paths:
        print(f"Evaluating {model_path}")
        print("-"*50)
        
        for test_set in sents_test_sets:
            if generate_outputs:
                print(f"====>> Generating outputs using {model_path.split('/')[2]} on {test_set.split('/')[-1]}.........")
            src_file_path = f"{test_set}/test.src"
            tgt_file_path = f"{test_set}/test.tgt"
            out_path = f"outputs/{trial_name}/{model_path.split('/')[2]}_on_{test_set.split('/')[-1]}"
            model_path_arg = '/'.join(model_path.split('/')[:-1])
            checkpoint_arg = model_path.split('/')[-1]
            rougeL = None
            if generate_outputs:
                output = subprocess.check_output(f"""python -m ats_models.inference_mbart \
                                        --model_path {model_path_arg} \
                                        --checkpoint {checkpoint_arg} \
                                        --tokenizer {model_path_arg} \
                                        --test_source {src_file_path} \
                                        --test_target {tgt_file_path} \
                                        --src_lang de_DE \
                                        --tgt_lang de_SI \
                                        --max_input_len 256 \
                                        --max_output_len 256 \
                                        --batch_size 16 \
                                        --num_workers 20 \
                                        --accelerator gpu \
                                        --devices 0 \
                                        --beam_size {beam_size} \
                                        --progress_bar_refresh_rate 1 \
                                        --translation {out_path} """, shell=True, text=True)
                
                
                for line in output.split('\n')[-10:]:
                    if 'rougeL:' in line:
                        rougeL = float(line.split('[')[-1][:-2])
                
                assert rougeL is not None, f"Could not find rougeL in {output} for {model_path} on {test_set}"
                
                print("="*50)
                print(f"Finished generating outputs using {model_path.split('/')[2]} on {test_set.split('/')[-1]} successfully.")
                print("="*50)
            
            # evaluate the outputs using easse-de
            
            print(f"====>> Evaluating generated outputs of {model_path.split('/')[2]} on {test_set.split('/')[-1]}.........")
            remove_lang_tags(out_path)
            output = subprocess.check_output(f"""easse evaluate \
                            -lang de \
                            -il sentence-level \
                            -m bleu,sari,bertscore,fre_sent,fre_corpus \
                            -t custom \
                            --orig_sents_path {src_file_path} \
                            --refs_sents_paths {tgt_file_path} \
                            {analys} -i {out_path} """, shell=True, text=True)
            
            metrics_str = output.split('\n')[-2]
            metrics_dict = ast.literal_eval(metrics_str)
            if anlaysis:
                quality_estimation = metrics_dict['quality_estimation']
            
            metrics_dict['compression_ratio'] = quality_estimation['Compression ratio'] if anlaysis else " "
            metrics_dict['sentence_splits'] = quality_estimation['Sentence splits'] if anlaysis else " "
            metrics_dict['levenshtein_similarity'] = quality_estimation['Levenshtein similarity'] if anlaysis else " "
            metrics_dict['exact_copies'] = quality_estimation['Exact copies'] if anlaysis else " "
            metrics_dict['additions_proportion'] = quality_estimation['Additions proportion'] if anlaysis else " "
            metrics_dict['deletions_proportion'] = quality_estimation['Deletions proportion'] if anlaysis else " "
            metrics_dict['lexical_complexity_score'] = quality_estimation['Lexical complexity score'] if anlaysis else " "
            
            metrics_dict['model'] = ' '.join(model_path.split('/')[2].split('_')[:2])
            metrics_dict['train_set'] = ' '.join(model_path.split('/')[2].split('_')[3:])
            metrics_dict['test_set'] = test_set.split('/')[-1]
            if old_csv is not None:
                rougeL = old_csv[(old_csv['model'] == metrics_dict['model']) & (old_csv['train_set'] == metrics_dict['train_set']) & (old_csv['test_set'] == metrics_dict['test_set'])]['rougeL'].values[0]
            metrics_dict['rougeL'] = rougeL
            metrics_dict['level'] = 'sentence'
            
            df_rows.append(metrics_dict)         
            
            print("="*50)
            print("="*50)
            print(f"Finished Evaluating {model_path} on {test_set} successfully.")
            print("="*50)
            print("="*50)
    
    df = pd.DataFrame(df_rows)
    df = df[['model',
             'train_set',
             'test_set', 
             'level', 
             'rougeL', 
             'bleu', 
             'sari', 
             'bertscore_precision', 
             'bertscore_recall', 
             'bertscore_f1', 
             'sent_FRE', 
             'corpus_FRE',
             'compression_ratio',
             'sentence_splits',
             'levenshtein_similarity',
             'exact_copies',
             'additions_proportion',
             'deletions_proportion',
             'lexical_complexity_score'
             ]]
    
    print("="*50)
    print("="*50)
    print("="*50)
    print("="*50)
    
    print('Saving evaluation metrics to csv file...')
    
    df.to_csv(f"eval_metrics/{trial_name}.csv", index=False)
    
    print("="*50)
    print("="*50)
    print("="*50)
    print("="*50)
    
    print('Done!')
        
    return df

def get_models_paths(trial_name):
    docs_models_paths = []
    sents_models_paths = []
    
    finetuned_checkpoints = os.listdir(f"finetuned_checkpoints/{trial_name}")
    
    for checkpoint in finetuned_checkpoints:
        if 'mbart' in checkpoint: # only mbart checkpoints
            # get the best checkpoint name
            with open(f"finetuned_checkpoints/{trial_name}/{checkpoint}/training_logs", "r") as f:
                lines = f.readlines()
                checkpoint_name = None
                for line in lines:
                    if 'Training ended. Best checkpoint' in line:
                        checkpoint_name = line.split('/')[-1].strip()[:-1]
                assert checkpoint_name is not None, f"Could not find best checkpoint in {checkpoint} training logs."
            
            model_path = f"finetuned_checkpoints/{trial_name}/{checkpoint}/{checkpoint_name}"
            
            assert os.path.exists(model_path), f"Could not find model file at {model_path}"
            if 'long' in checkpoint: # document level models
                docs_models_paths.append(model_path)
            else: # sentence level models
                sents_models_paths.append(model_path)
    
    
    return docs_models_paths, sents_models_paths

def remove_lang_tags(path):
    clean_lines = []
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            clean_lines.append(line.replace('de_DE', '').replace('de_SI', '').strip())
    
    with open(path, "w") as f:
        for line in clean_lines:
            f.write(line + '\n')
            