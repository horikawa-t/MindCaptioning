# filename: mcap_evaluation.py
# source activate mcap_demo
# 
# This script is written to evalaute sentence generation results from brain-decoded features in
#   Horikawa, T. (2024) Mind captioning: Evolving descriptive text of mental content from human brain activity. bioRxiv. 
# 
#  written by Tomoyasu Horikawa horikawa.t@gmail.com 2024/05/13
# 

# set path to MindCaptioning directory
rootPath = './'

import os
import sys
import torch
import numpy as np
import random
import scipy.io  as sio
import time
import h5py
import itertools as itt
import csv
import gc
import json
import os
import sys
import torch
import time
from bert_score import BERTScorer
# add path to code
sys.path.append(rootPath+'code/python/util/')
from thutil4 import getFN, getDN, setdir, fix_seed,randsample
import mcap_utils_demo as mu

gpu_use = 1
if gpu_use:
    gpu_id = '0'
    print('Start script: gpu device:%s' %(gpu_id))
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = ""
device = "cuda" if torch.cuda.is_available() and gpu_use == 1 else "cpu"
print('gpu availability:%d'%(torch.cuda.is_available()))
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

# general settings
save_log = 1
dele = 0 # delete option: if 1, remove unfinished log

savdir            = rootPath + 'res/text_generation/'
decfeat_dir       = rootPath + 'res/decoding/'
LMmodeldir        = rootPath + 'data/model/'
normparam_dat_dir = rootPath + 'data/feature/norm_param/'
capdata_dir       = rootPath + 'data/caption/'

# set proxy if necessary
proxies = {
    "http": "",
    "https": "",
}

# set data types
dataTypes = ['testPerception','testImagery'] # ['testPerception','testImagery','trainPerception']
capType = 'ck20'

roiTypes = ['WB','WBnoLang','WBnoSem','WBnoVis','Lang']

# subject settings
sbjs = ['S1','S2','S3','S4','S5','S6']

# setting model
# select MLM from ['bert-base-cased','bert-base-uncased','bert-large-cased','bert-large-uncased','bert-large-uncased-whole-word-masking','bert-large-cased-whole-word-masking','roberta-base','roberta-large','deberta-large-feedback']
# you can test untrained model by addiing "_untrained" (e.g., 'roberta-large_untrained')
MLMType = 'roberta-large' 

# select LM for feature extraction from ['bert-base-uncased','bert-large-uncased','bert-base-cased','bert-large-cased','bert-large-uncased-whole-word-masking','bert-large-cased-whole-word-masking','openai-gpt','gpt2','gpt2-medium','gpt2-large','gpt2-xl','xlnet-base-cased','xlnet-large-cased','roberta-base','roberta-large','distilbert-base-uncased','distilbert-base-cased','distilgpt2','albert-base-v1','albert-large-v1','albert-xlarge-v1','albert-xxlarge-v1','albert-base-v2','albert-large-v2','albert-xlarge-v2','albert-xxlarge-v2','t5-small','t5-base','t5-large','bart-base','bart-large','ctrl','xlm-mlm-17-1280','xlm-mlm-100-1280','electra','xlm-roberta-base','xlm-roberta-large','clip_l','sgpt','deberta-base','deberta-large','deberta-xlarge']
LMType = 'deberta-large'

# load pre-trained masked language model
tokenizer, model = mu.load_mlm_model(LMmodeldir, MLMType, proxies, device)
# load feature computation model
tokenizer_lm, model_lm, nlayers = mu.load_lm_model(LMmodeldir, LMType, proxies, device)
# load bertscore model
scorer_base = BERTScorer(lang="en", all_layers = False, rescale_with_baseline=True)
# set evaluation mode
model.eval(),model_lm.eval()

# prepre skip tokens, if any
skip_token_ids_mlm = mu.set_skip_token_ids(tokenizer, speficied_skip_tokens=[], include_special_token=True)
skip_token_ids_lm = mu.set_skip_token_ids(tokenizer_lm, speficied_skip_tokens=[], include_special_token=True)


# set parameters
params = {
    'nItr': 100,
    'metricType': 'corr',
    'do_norm': 1,
    'beamwidth': 5,
    'nMaskCands': 5,
    'nMaskPerSentence': 2,
    'nGram4Mask': 3,
    'multiMaskType':'forward_seq',
    'maskingUnitType':'token',
    'add_insert_mask': 1,
    'mLayerType': 'vstack',
    'optimal_th': 0.001,
    'topk': 5,
    'max_batch_samp': 200,
    'length_penalty_type':'token',
    'length_penalty_w': 0.10,
    'mlmscoreType': 'modified',
    'mlm_sampling_type': 'sampling',
    'mlms_fix_weight': 0,
    'nMax_MLMs_cands':5000,
    'do_reflesh': 1,
    'reflesh_th': [10,0.1,5,0.00],
    'add_mask_removal': False,
    'layerIdx': range(0,nlayers),
    'device': device,
}
nRep1shot = 5

# load normalization parameters
normparam_path = f"{normparam_dat_dir}/{LMType}/"
feat_mu_all, feat_sd_all = mu.prepare_norm_params(normparam_path, nlayers, device=device)

# load caption data
caps_data = mu.load_caption_data(capdata_dir, [capType])
nCapEach = 20

# initialize
start = time.time()

# set evaluation metrics
eval_metrics = ['BLEU','METEOR','ROUGE-L','CIDEr','BERTscore']

## start analysis
for dataType, roiType, sbj in itt.product(dataTypes,roiTypes,sbjs):
    
    # assign # of samples
    if dataType.startswith(('test')):
        decsampIdx = range(0,72)
    elif dataType.startswith(('train')):
        decsampIdx = range(0,2108)
    
    chks = np.zeros(len(decsampIdx))
    for decsampidx in decsampIdx:

        # prepare target features info
        decfeat_path = f"{decfeat_dir}/{dataType}/{LMType}/{sbj}/{roiType}/"
        videoidx, videoIdx, skipflag = mu.prepare_label_parameters(decsampidx, decfeat_path, device)
        caps_each = caps_data[videoidx*nCapEach:((videoidx+1)*nCapEach)]

        # prepare save filenames
        saveFnameChk = f"{savdir}/{dataType}/mlm_{MLMType}/lm_{LMType}/{sbj}/{roiType}/log/log_summary_log.txt"
        saveFname    = f"{savdir}/{dataType}/mlm_{MLMType}/lm_{LMType}/{sbj}/{roiType}/res/res_summary.mat"
        mu.setdir(os.path.dirname(saveFnameChk))
        mu.setdir(os.path.dirname(saveFname))

    if os.path.isfile(saveFnameChk):
        print('Skip:%s'%(saveFnameChk))
        if dele and os.path.isfile(saveFname) == False:
            print('Delete:' + saveFnameChk)
            try:os.remove(saveFnameChk)
            except:print('Failed to delete:' + saveFnameChk)             
    elif dele == 0:
        mu.save_logfile(saveFnameChk) if save_log else print('Skip saving:%s'%(saveFnameChk))
        print(f'Start evaluation')
        start = time.time()

        print('Load prediction results...[t=%.5f]'%(time.time() - start))
        best_cands,videoIdx_all = [],[]
        skpflag = 0
        for decsampidx in decsampIdx:
            dataFname    = f"{savdir}/{dataType}/mlm_{MLMType}/lm_{LMType}/{sbj}/{roiType}/res/res_samp{decsampidx+1:04d}.mat"
            try:
                restmp = sio.loadmat(dataFname,variable_names=['best_cands','best_cands_len','videoidx'])
            except:
                skipflag = 1
                break
            videoIdx_all.append(restmp['videoidx'][0][0]-1)
            best_cands.append(tokenizer.decode(tokenizer.encode(restmp['best_cands'][-1],add_special_tokens=False)[:restmp['best_cands_len'][0][-1]]))

        if skipflag:
            os.remove(saveFnameChk)
            print('Full data was not available. Stop summarization for this process.')
            continue

        # perform feature correlation based evaluation
        print('Start feature correlation based evaluation')
        # compute feature and scores for preds
        feat_preds, inputs = mu.compute_sentence_feature_patterns_wrapper(best_cands, model_lm, tokenizer_lm, skip_token_ids=skip_token_ids_lm, feat_mu_all=feat_mu_all, feat_sd_all=feat_sd_all, do_norm=True, device=device, layerIdx=range(nlayers), max_batch_samp=20)

        # compute feature and scores for all cands
        scores_gen2ref_all = []
        max_batch_samp_eval = 1000
        nbatch, modu, n_add = mu.make_minibatch_params(len(caps_data), max_batch_samp_eval)
        for bi in range(nbatch + n_add):
            sidx = bi * max_batch_samp_eval
            eidx = (bi + 1) * max_batch_samp_eval if bi < nbatch else None
            print(f'{bi+1}/{nbatch + n_add} [t={time.time() - start:.5f}]')

            # compute feature and scores for subsets of cands
            feat_cand_evals = mu.compute_sentence_feature_patterns_wrapper(caps_data[sidx:eidx], model_lm, tokenizer_lm, skip_token_ids=skip_token_ids_lm, feat_mu_all=feat_mu_all, feat_sd_all=feat_sd_all, do_norm=True, device=device, layerIdx=range(nlayers), max_batch_samp=10)[0]
            scores_gen2ref_all.append([mu.compute_score(tokenizer_lm, [], feat_cand_eval, feat_preds, params['mLayerType'], params['metricType'], skip_token_ids=skip_token_ids_lm)[0] for feat_cand_eval in feat_cand_evals])
            del feat_cand_evals

        # reshape
        scores_gen2ref_all = torch.vstack(mu.flatten_list(scores_gen2ref_all))

        # correlation
        scores_gen2ref_eval_maxs = torch.vstack([mu.step_summarize_scores(scores_gen2ref_all[:,i],nCapEach, summaryType='max', device=device) for i in range(len(videoIdx_all))]).to(device)
        scores_gen2ref_eval_means = torch.vstack([mu.step_summarize_scores(scores_gen2ref_all[:,i],nCapEach, summaryType='mean', device=device) for i in range(len(videoIdx_all))]).to(device)
        scores_gen2ref_eval_true_maxs = torch.tensor([scores_gen2ref_eval_maxs[i,videoidx] for i,videoidx in enumerate(videoIdx_all)])
        scores_gen2ref_eval_true_means = torch.tensor([scores_gen2ref_eval_means[i,videoidx] for i,videoidx in enumerate(videoIdx_all)])
        scores_gen2ref_eval_false_maxs = torch.vstack([torch.concatenate((scores_gen2ref_eval_maxs[i,:videoidx],scores_gen2ref_eval_maxs[i,videoidx+1:])) for i,videoidx in enumerate(videoIdx_all)])
        scores_gen2ref_eval_false_means = torch.vstack([torch.concatenate((scores_gen2ref_eval_means[i,:videoidx],scores_gen2ref_eval_means[i,videoidx+1:])) for i,videoidx in enumerate(videoIdx_all)])

        idens_gen2ref_eval_max = torch.tensor([mu.compute_identifications_scores(scores_gen2ref_eval_true_maxs[i], scores_gen2ref_eval_false_maxs[i,:], summaryType='max', device=device) for i in range(len(videoIdx_all))]).to(device)
        idens_gen2ref_eval_mean = torch.tensor([mu.compute_identifications_scores(scores_gen2ref_eval_true_means[i], scores_gen2ref_eval_false_means[i,:],  summaryType='mean', device=device) for i in range(len(videoIdx_all))]).to(device)
        print('gen2falseRef[max:r = %.4f; mean:r = %.4f] [t=%.5f]'%(torch.mean(scores_gen2ref_eval_true_maxs),torch.mean(scores_gen2ref_eval_true_means),time.time() - start))
        print('gen2falseRef[max:cr = %.4f%%; mean:cr = %.4f%%] [t=%.5f]'%(torch.mean(idens_gen2ref_eval_max)*100,torch.mean(idens_gen2ref_eval_mean)*100,time.time() - start))

        # perform nlp metric evaluation
        print('Start NLP based evaluation')
        SCORES = mu.evaluate_nlp_metric_wrapper2(tokenizer=tokenizer, caps_preds=best_cands, caps_alls=caps_data, labels=videoIdx_all, eval_metrics=eval_metrics, nCapEach=nCapEach, scorer_base=scorer_base, device=device)
        res = {'best_cands':best_cands,
            'videoIdx_all':videoIdx_all,
            'scores_gen2ref_all':np.array(scores_gen2ref_all.cpu().detach()),
            'scores_gen2ref_eval_maxs':np.array(scores_gen2ref_eval_maxs.cpu().detach()),
            'scores_gen2ref_eval_means':np.array(scores_gen2ref_eval_means.cpu().detach()),
            'scores_gen2ref_eval_true_maxs':np.array(scores_gen2ref_eval_true_maxs.cpu().detach()),
            'scores_gen2ref_eval_true_means':np.array(scores_gen2ref_eval_true_means.cpu().detach()),
            'scores_gen2ref_eval_false_maxs':np.array(scores_gen2ref_eval_false_maxs.cpu().detach()),
            'scores_gen2ref_eval_false_means':np.array(scores_gen2ref_eval_false_means.cpu().detach()),
            'idens_gen2ref_eval_max':np.array(idens_gen2ref_eval_max.cpu().detach()),
            'idens_gen2ref_eval_mean':np.array(idens_gen2ref_eval_mean.cpu().detach()),
            'bleu_scores_true_refs':SCORES['BLEU']['scores_true_refs'],
            'bleu_scores_false_refs':SCORES['BLEU']['scores_false_refs'],
            'bleu_idens_scores':SCORES['BLEU']['idens_scores'],
            'meteor_scores_true_refs':SCORES['METEOR']['scores_true_refs'],
            'meteor_scores_false_refs':SCORES['METEOR']['scores_false_refs'],
            'meteor_idens_scores':SCORES['METEOR']['idens_scores'],
            'rouge_scores_true_refs':SCORES['ROUGE-L']['scores_true_refs'],
            'rouge_scores_false_refs':SCORES['ROUGE-L']['scores_false_refs'],
            'rouge_idens_scores':SCORES['ROUGE-L']['idens_scores'],
            'cider_scores_true_refs':SCORES['CIDEr']['scores_true_refs'],
            'cider_scores_false_refs':SCORES['CIDEr']['scores_false_refs'],
            'cider_idens_scores':SCORES['CIDEr']['idens_scores'],
            'F1_scores_true_refs':SCORES['F1']['scores_true_refs'],
            'F1_scores_false_refs_org':SCORES['F1']['scores_false_refs_org'],
            'F1_scores_false_refs_max':SCORES['F1']['scores_false_refs_max'],
            'F1_scores_false_refs_mean':SCORES['F1']['scores_false_refs_mean'],
            'F1_idens_scores_max':SCORES['F1']['idens_scores_max'],
            'F1_idens_scores_mean':SCORES['F1']['idens_scores_mean'],
            'R_scores_true_refs':SCORES['R']['scores_true_refs'],
            'R_scores_false_refs_org':SCORES['R']['scores_false_refs_org'],
            'R_scores_false_refs_max':SCORES['R']['scores_false_refs_max'],
            'R_scores_false_refs_mean':SCORES['R']['scores_false_refs_mean'],
            'R_idens_scores_max':SCORES['R']['idens_scores_max'],
            'R_idens_scores_mean':SCORES['R']['idens_scores_mean'],
            'P_scores_true_refs':SCORES['P']['scores_true_refs'],
            'P_scores_false_refs_org':SCORES['P']['scores_false_refs_org'],
            'P_scores_false_refs_max':SCORES['P']['scores_false_refs_max'],
            'P_scores_false_refs_mean':SCORES['P']['scores_false_refs_mean'],
            'P_idens_scores_max':SCORES['P']['idens_scores_max'],
            'P_idens_scores_mean':SCORES['P']['idens_scores_mean'],
            }
        if save_log and dele == 0:
            print('Save:%s'%(saveFname))
            sio.savemat(saveFname, res)
            print('Saved.')



print('Done.')

