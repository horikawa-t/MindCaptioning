# filename: mcap_analysis.py
# source activate mcap_demo
# 
# This script is written to perform sentence generation analysis from brain-decoded features
#   Horikawa, T. (2024) Mind captioning: Evolving descriptive text of mental content from human brain activity. bioRxiv. 
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
        saveFnameChk = f"{savdir}/{dataType}/mlm_{MLMType}/lm_{LMType}/{sbj}/{roiType}/log/log_samp{decsampidx+1:04d}_log.txt"
        saveFname    = f"{savdir}/{dataType}/mlm_{MLMType}/lm_{LMType}/{sbj}/{roiType}/res/res_samp{decsampidx+1:04d}.mat"
        mu.setdir(os.path.dirname(saveFnameChk))
        mu.setdir(os.path.dirname(saveFname))


        if os.path.isfile(saveFnameChk) or os.path.isfile(saveFname):
            print('Skip:%s'%(saveFnameChk))
            if os.path.isfile(saveFname):
                chks[decsampidx] = 1                    
            if dele and os.path.isfile(saveFname) == False:
                print('Delete:' + saveFnameChk)
                try:os.remove(saveFnameChk)
                except:print('Failed to delete:' + saveFnameChk)             
        elif dele == 0:
            mu.save_logfile(saveFnameChk) if save_log else print('Skip saving:%s'%(saveFnameChk))

            # initialization 1
            seeds,best_cands_all,scores_alls,scores_eval_alls = [],[],[],[]
            start = time.time()
            initial_state = tokenizer.unk_token

            print('\nStart from: %s' %(initial_state))
            target_sentence = 'brain-decoded feature'
            print('brain sample:[%s][samp%d:vid%d][%s:%s]' %(roiType,decsampidx+1,videoidx+1,sbj,dataType))
            print('Parameters============')
            print('MLM model: %s' %(MLMType))
            print('LM feature type: %s [L:%d to %d]' %(LMType,params['layerIdx'][0]+1,params['layerIdx'][-1]+1))
            print('beamwidth:%d' %(params['beamwidth']))
            print('nMaskCands:%d' %(params['nMaskCands']))
            print('n sampling:%d by %s' %(params['topk'],params['mlm_sampling_type']))
            print('multi-mask fill type: %s' %(params['multiMaskType']))
            print('n iterations:%d' %(params['nItr']))
            print('length penalty with weight:%.2f' %(params['length_penalty_w']))

            # load tentative res
            for nrepitr in range(nRep1shot): # # of repes for best seletion: this help reduce evaluation time

                print('======================')
                print('Start %d/%d generation [video:%d]'%(nrepitr+1,nRep1shot,videoidx+1))
                # set seed
                seeds.append(int(time.time()*10000)%(2**32)) # set&keep seed for the reproducability
                mu.fix_seed(seeds[-1])
                print('seed:%d'%(seeds[-1]))
                print('======================\n')

                # prepare target feature
                feat_target = mu.prepare_feature_data(decsampidx, decfeat_path, params)[0]


                # text optimization steps
                try:
                    best_cands, scores_all, scores_eval_all = mu.text_optimization_steps(feat_target, feat_mu_all, feat_sd_all, model, tokenizer, skip_token_ids_mlm, model_lm, tokenizer_lm, skip_token_ids_lm, params, device)
                except:
                    print('Failed to finish analysis. Delete:%s'%(saveFnameChk))
                    try:os.remove(saveFnameChk)
                    except:print('Failed to delete:' + saveFnameChk)             
                    skipflag = 1
                    break

                print('Final output: %s'%(best_cands[-1]))
                print('Score[gen2bdec]: %.4f'%(scores_all[-1]))
                scores_all = torch.tensor(scores_all).to(device)
                scores_eval_all = torch.tensor(scores_eval_all).to(device)

                # keep each sub rep
                best_cands_all.append(best_cands)
                scores_alls.append(scores_all)
                scores_eval_alls.append(scores_eval_all)

                print('Processing:%s\n'%(saveFnameChk))

            if skipflag:
                continue

            print('======================')
            print('End %d/%d generation [video:%d]'%(nrepitr+1,nRep1shot,videoidx+1))
            print('target sentence:%s'%(target_sentence))
            # set seed
            seeds.append(int(time.time()*10000)%(2**32)) # set&keep seed for the reproducability
            print('======================\n')

            # summary sub candidates
            print('Sub generated candidates:')
            for bc, sa, se in zip(best_cands_all,scores_alls,scores_eval_alls):
                print('[%.4f, %.4f]:%s'%(sa[-1],se[-1],bc[-1]))

            # further process best sub results
            mxval, bestSubRepIdx = torch.max(torch.vstack(scores_eval_alls)[:,-1],dim=0)
            best_cands = best_cands_all[bestSubRepIdx]
            scores_all = scores_alls[bestSubRepIdx]
            scores_eval_all = scores_eval_alls[bestSubRepIdx]

            print('Best output: %s'%(best_cands[-1]))
            print('Score[gen2bdec]: %.4f\n'%(scores_all[-1]))

            # compute prediction length
            best_cands_len = [len([tokenizer.decode(e) for e in tokenizer.encode(bc,add_special_tokens=False)]) for bc in best_cands]
            best_cands_len_all = [[len([tokenizer.decode(e) for e in tokenizer.encode(bc,add_special_tokens=False)]) for bc in best_cand_temp] for best_cand_temp in best_cands_all]


            # summarize results ========
            # get unique bests
            unique_bests, uni2org_idx_tmp = mu.get_unique_list(best_cands)
            lastIdx = uni2org_idx_tmp[params['nItr']][0]
            uni2org_idx = [i[0] for c, i in uni2org_idx_tmp.items()]
            nUniBests = len(unique_bests)

            # compute generated sentence features for a unique set
            print('Prepare features for evaluation')
            feat_gen_bests = mu.compute_sentence_feature_patterns_wrapper(unique_bests, model_lm, tokenizer_lm, skip_token_ids=skip_token_ids_lm, do_norm=params['do_norm'], feat_mu_all=feat_mu_all, feat_sd_all=feat_sd_all, device=device, layerIdx=params['layerIdx'], max_batch_samp=params['max_batch_samp'])[0]
            # compute reference sentence features
            feat_refs = mu.compute_sentence_feature_patterns_wrapper(caps_each, model_lm, tokenizer_lm, skip_token_ids=skip_token_ids_lm, do_norm=params['do_norm'], feat_mu_all=feat_mu_all, feat_sd_all=feat_sd_all, device=device, layerIdx=params['layerIdx'], max_batch_samp=params['max_batch_samp'])[0]

            # compute correlation between features of reference captions and generated captions
            scores_best = torch.vstack([mu.compute_score(tokenizer_lm, [], feat_best, feat_refs, mLayerType=params['mLayerType'], metricType=params['metricType'], skip_token_ids=skip_token_ids_lm)[0] for feat_best in feat_gen_bests]).to(device)
            bestrefidx = torch.argmax(scores_best[lastIdx])
            print('Best Ref index [gen]:%d:%s'%(bestrefidx,caps_each[bestrefidx]))
            print('gen2ref[max:last]:r = %.4f'%(torch.max(scores_best[lastIdx])))
            print('End computing reference-to-generated similarity [t=%.5f]'%(time.time() - start))

            # preserve tentative results
            res = {}
            res['initial_state'] = initial_state
            res['target_sentence'] = target_sentence
            res['decsampidx'] = decsampidx
            res['seeds'] = seeds
            res['best_cands'] = best_cands
            res['best_cands_all'] = best_cands_all
            res['scores_all'] = np.array(scores_all.cpu().detach())
            res['scores_eval_all'] = np.array(scores_eval_all.cpu().detach())
            res['scores_alls'] = np.array(torch.vstack(scores_alls).cpu().detach())
            res['scores_eval_alls'] = np.array(torch.vstack(scores_eval_alls).cpu().detach())
            res['best_cands_len'] = np.array(best_cands_len)
            res['best_cands_len_all'] = np.array(best_cands_len_all)
            res['bestSubRepIdx'] = np.array(bestSubRepIdx.cpu().detach())
            res['videoidx'] = videoidx+1
            res['scores_best'] = np.array(scores_all.cpu().detach())
            res['bestrefidx'] = np.array(bestrefidx.cpu().detach())

            if save_log and dele == 0:
                print('Save:%s'%(saveFname))
                sio.savemat(saveFname, res)




print('Done.')




