# mcap_utils_demo.py

import torch
import random
import numpy as np
import gc

def save_logfile(fname):
    print('Save:%s'%(fname))
    ff = open(fname,'w')
    ff.write(' ')
    ff.close()

def convert_index_and_captions_to_dict(index, captions):
    gts = {}

    for i, caption in enumerate(captions):
        img_id = index[i]
        if img_id not in gts:
            gts[img_id] = []
        gts[img_id].append(caption)

    return gts


# torch function
def shuffle_word_sequence(word_sequence, n_shuffle=1000, keep_full_stop=True):
    word_seq_sh_all = []
    max_ntrials = n_shuffle*10 # to avoid infinite loop
    
    # Function to perform word shuffling
    def shuffle_words(word_sequence):
        word_list = word_sequence.split(' ')
        shuffled_words = ' '.join(random.sample(word_list, len(word_list)))
        return shuffled_words
    
    nfailcnt = 0
    while len(word_seq_sh_all) < n_shuffle:
        if keep_full_stop and word_sequence[-1] == '.':
            word_seq_base = word_sequence[0:-1]
        else:
            word_seq_base = word_sequence
        
        word_seq_sh = shuffle_words(word_seq_base)
        
        while word_seq_sh == word_seq_base:
            word_seq_sh = shuffle_words(word_seq_sh)
        
        if keep_full_stop and word_sequence[-1] == '.':
            word_seq_sh = word_seq_sh + '.'
        
        word_seq_sh_all.append(word_seq_sh)
        if len(list(set(word_seq_sh_all))) < len(word_seq_sh_all):
            nfailcnt += 1
            if nfailcnt >= max_ntrials:
                word_seq_sh_all = list(set(word_seq_sh_all))
                print('Only %d shuffled sequences were generated.'%(len(word_seq_sh_all)))
                return word_seq_sh_all, len(word_seq_sh_all)
        word_seq_sh_all = list(set(word_seq_sh_all))
    
    return word_seq_sh_all, n_shuffle


def remove_diagonal_elements(matrix, reduce='col'):
    """
    Remove the diagonal elements from a matrix and reduce either rows or columns.

    Args:
        matrix (torch.Tensor): The input matrix.
        reduce (str): Reduction direction ('row' or 'col'/'column').

    Returns:
        torch.Tensor: The reduced matrix.

    Raises:
        ValueError: If an invalid reduce option is provided.

    # Example usage
    matrix = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    # Remove diagonal elements and reduce by columns
    reduced_matrix = remove_diagonal_elements(matrix, reduce='col')

    print("Original Matrix:")
    print(matrix)

    print("\nReduced Matrix:")
    print(reduced_matrix)

    """
    n = matrix.size(0)
    if reduce in ['col', 'column']:
        reduced_matrix = torch.zeros((n, n-1))
        for i in range(n):
            reduced_matrix[i] = torch.cat((matrix[i, :i], matrix[i, i+1:]))
    elif reduce == 'row':
        reduced_matrix = torch.zeros((n-1, n))
        for i in range(n):
            reduced_matrix[:, i] = torch.cat((matrix[:i, i], matrix[i+1:, i]))
    else:
        raise ValueError("Invalid reduce option. Choose 'row' or 'col'/'column'.")
    
    return reduced_matrix


def get_unique_list(list_org):
    """
    Get a list of unique elements from the input list and their corresponding indices in the original list.

    Parameters:
        list_org (list): The input list from which unique elements and indices will be extracted.

    Returns:
        tuple: A tuple containing:
            - list: A new list containing only the unique elements from the input list.
            - dict: A dictionary where keys are indices of unique elements in the new list,
                    and values are lists of indices in the original list corresponding to each unique element.
    
    Example:
        list_org = [1, 2, 3, 2, 4, 5, 4]
        unique_list, index_dict = get_unique_list(list_org)
        print(unique_list)  # Output: [1, 2, 3, 4, 5]
        print(index_dict)   # Output: {0: [0], 1: [1, 3], 2: [2], 3: [4, 6], 4: [5]}
    """
    uni2org_idx = {ix: [i for i, x in enumerate(set(list_org)) if x == item] for ix, item in enumerate(list_org)}
    return list(set(list_org)), uni2org_idx

def flatten_list(nested_list):
    """
    Flattens a nested list into a single list.

    Args:
        nested_list (list): The nested list to be flattened.

    Returns:
        list: The flattened list.
    """
    flattened_list = [item for sublist in nested_list for item in sublist]
    return flattened_list

def flatten_nested_list(nested_list): # 20230920 todo check the compatibility with flatten_list
    """
    Flattens a nested list into a single list.

    Args:
        nested_list (list): The nested list to be flattened.

    Returns:
        flattened_list: The flattened list.
    """
    flattened_list = []
    for item in nested_list:
        if isinstance(item, list):
            flattened_list.extend(flatten_nested_list(item))
        else:
            flattened_list.append(item)
    return flattened_list


def z_normalize(data):
    device = data.device
    dtype = data.dtype
    mean_data = torch.mean(data)
    std_data = torch.std(data)
    
    # Ensure the standard deviation is not zero
    epsilon = torch.tensor(1e-8, dtype=dtype, device=device)
    std_data = torch.max(std_data, epsilon)
    
    normalized_data = (data - mean_data) / std_data
    
    return normalized_data, mean_data, std_data

def corr2_coeff(A, B):
    # from https://stackoverflow.com/questions/30143417/computing-the-correlation-coefficient-between-two-multi-dimensional-arrays/30143754#30143754
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1)
    ssB = (B_mB**2).sum(1)

    # Finally get corr coeff
    return torch.matmul(A_mA, B_mB.T) / torch.sqrt(torch.matmul(ssA[:, None],ssB[None]))

# sampling function
def sample_without_replacement(logits, K):
    """
    Samples K elements without replacement from a distribution defined by logits.

    Args:
        logits (torch.Tensor): A 1D tensor of logits.
        K (int): The number of elements to sample.

    Returns:
        torch.Tensor: A 1D tensor of indices representing the sampled elements.

    Adapted from: https://github.com/tensorflow/tensorflow/issues/9260#issuecomment-437875125
    """
    z = -torch.log(-torch.log(torch.rand(logits.shape))).to(logits.device)
    _, indices = torch.topk(logits + z, K)
    return indices

def tril_flatten(tril):
    N = tril.size(-1)
    #indicies = torch.tril_indices(N, N) # with diagonal
    indicies = torch.tril_indices(N, N,-1) # only off-diagonal
    indicies = N * indicies[0] + indicies[1]
    return tril.flatten(-2)[..., indicies]

def count_special_tokens(input_ids, skip_token_ids):
    nspecial_tokens = [torch.sum(torch.where(input_ids == skip_token_id, 1, 0), dim=1) for skip_token_id in skip_token_ids]
    return torch.sum(torch.vstack(nspecial_tokens),dim=0)

def compute_score(tokenizer, inputs, featureset1, featureset2, mLayerType, metricType='corr', skip_token_ids=[], length_penalty_type='token', length_penalty_w=0):
    """
    Compute the correlation-based score between two feature sets.
    
    Arguments:
    - tokenizer (Tokenizer): An instance of the Tokenizer class from the Hugging Face transformers library.
    - inputs: Input data for computing length for length penalty
    - featureset1 (list: nlayers): First feature set.
    - featureset2 (list: nSamples has nlayers): Second feature set.
    - mLayerType: Layer type for combining features ('hstack' or 'vstack').
    - metricType: Type of metric to use ('corr' for correlation).
    - skip_token_ids (list): A list of token IDs to skip during sampling. 
    - length_penalty_type: Unit of length (default='token' or 'word')
    - length_penalty_w: weight for the length penalty to the score.
    
    Returns:
    - scores: Final scores.
    - scores_reg: Regularized scores.
    """
    if metricType == 'corr':
        # Compute correlation between feature sets with specified mLayerType
        if mLayerType == 'hstack':
            scores = corr2_coeff(torch.hstack(featureset1).view(1, -1), torch.vstack([torch.hstack(wm) for wm in featureset2]))[0]
        elif mLayerType == 'vstack':
            scores_all = []
            # Calculate r for each layer and average them across layers
            for litr in range(len(featureset1)):
                scores_all.append(corr2_coeff(featureset1[litr].view(1, -1), torch.vstack([wm[litr] for wm in featureset2])))
            scores = torch.mean(torch.vstack(scores_all), dim=0)

    # Length penalty
    if length_penalty_w != 0:        
        # count ntokens
        cands_len = inputs['input_ids'].shape[1] - count_special_tokens(inputs['input_ids'], skip_token_ids)
        if length_penalty_type == 'word': # subtract # of ##""
            nsubword = torch.tensor([' '.join(tokenizer.convert_ids_to_tokens(inpt)).count('##') for inpt in inputs['input_ids']]).to(device)
            cands_len = cands_len - nsubword


        # Avoid zero-division (e.g., single [UNK] token case)
        cands_len = torch.where(cands_len <= 0, 1, cands_len)
        scores_reg = scores / (cands_len.permute(*torch.arange(cands_len.ndim - 1, -1, -1)).clone().detach() ** length_penalty_w)
        scores_reg = torch.where(scores <= 0, scores, scores_reg) # cancel length penalty for cand with scores < 0
    else:
        scores_reg = scores
        cands_len = []

    return scores, scores_reg, cands_len


def make_minibatch_params(ncands: int, max_batch_samp: int) -> tuple:
    """
    Returns parameters for minibatching.
    
    Parameters:
    ncands (int): Total number of input candidates.
    max_batch_samp (int): Maximum number of samples in each minibatch.
    
    Returns:
    tuple: A tuple of three integers representing the number of batches, the modulus, 
           and the number to add to the batches, respectively.
    """
    nbatch = np.floor(ncands / max_batch_samp).astype('int')
    modu = ncands % max_batch_samp
    n_add = int(modu > 0)
    
    return nbatch, modu, n_add

def get_unique(x, dim=0):
    # from: https://github.com/pytorch/pytorch/issues/36748
    unique, inverse, counts = torch.unique(x, dim=dim, 
        sorted=True, return_inverse=True, return_counts=True)
    decimals = torch.arange(inverse.numel(), device=inverse.device) / inverse.numel()
    inv_sorted = (inverse+decimals).argsort()
    tot_counts = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)))[:-1]
    index = inv_sorted[tot_counts]
    return unique, inverse, counts, index

def step_summarize_scores(scores, nstep, summaryType='max',device='cpu'):
    nSample = int(len(scores)/nstep)
    if summaryType == 'max':
        summary_scores = torch.tensor([torch.max(scores[ix * nstep:(ix + 1) * nstep]) for ix in range(nSample)]).to(device)
    elif summaryType == 'mean':
        summary_scores = torch.tensor([torch.mean(scores[ix * nstep:(ix + 1) * nstep]) for ix in range(nSample)]).to(device)
    return summary_scores

def compute_identifications_scores(pred_scores, cand_scores, summaryType='max', device='cpu'):
    ncands = len(cand_scores)
    if summaryType == 'max':
        identifications = torch.sum(torch.max(pred_scores) > cand_scores) / ncands
    elif summaryType == 'mean':
        identifications = torch.sum(torch.mean(pred_scores) > cand_scores) / ncands
    return identifications


class OnlineMeanStdDev:
    '''
    [example]
    import numpy as np
    data = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])]
    oms = OnlineMeanStdDev()
    for x in data:
        oms.add_data_point(x)
        print("Mean:", oms.get_mean())
        print("Standard Deviation:", oms.get_stddev())

        print("Mean[np]:", np.mean(data[:ix],axis=0))
        print("Standard Deviation[np]:", np.std(data[:ix],axis=0))

    '''

    def __init__(self):
        self.n = 0
        self.mean = None
        self.std = None
        self.var = None
        self.unbiased = 0

    def add_data_point(self, x: np.ndarray):
        self.n += 1
        #print(self.n)
        if self.mean is None:
            self.mean = np.zeros(x.shape)
            self.std = np.zeros(x.shape)
            self.var = np.zeros(x.shape)

        delta = x - self.mean
        self.mean += delta / self.n
        var_org = self.std
        self.var += delta * (x - self.mean)
        
        self.std = np.sqrt(self.var / self.n)

        if self.unbiased and self.n > 1:
            self.std = np.sqrt(self.var / (self.n-1))
        else:
            self.std = np.sqrt(self.var / self.n)
    
    def get_mean(self):
        return self.mean
    
    def get_stddev(self):
        return self.std

    def get_var(self):
        return self.var




def construct_head_subword_mapping(tokenizer, device, inpids):
    """
    Construct head subword mapping of iputs using a given tokenizer.

    Args:
        tokenizer: The tokenizer associated with the model.
        device (torch.device): A device object specifying the device to be used (e.g. "cpu" or "cuda").
        inpids (list): List of input ids.

    Returns:
        head_subword_mapping (list): head subword mapping
    """
    subword_token_index = (torch.tensor([tok.count('##') for tok in tokenizer.convert_ids_to_tokens(inpids)])==1).to(device)*1
    head_token_index = torch.cat((subword_token_index, torch.tensor([0], dtype=torch.int, device=device)))[1:]
    head_subword_mapping = []

    for i in range(len(head_token_index)):
        if head_token_index[i] == 1:
            subword_indices = []
            for j, val in enumerate(subword_token_index[i+1:], start=i+1):
                if val == 1:
                    subword_indices.append(j)
                else:
                    break
            head_subword_mapping.append((i, subword_indices))
    subword_token_index = torch.where(subword_token_index==1)[0]
    head_token_index = subword_token_index-1
    return head_subword_mapping, subword_token_index, head_token_index





def compute_mlm_scores(model, tokenizer, device, txts, mlmscoreType='original', max_batch_samp=200):
    """
    Compute MLM scores for a list of texts using a given model and tokenizer.

    Args:
        model (torch.nn.Module): The MLM model.
        tokenizer: The tokenizer associated with the model.
        device (torch.device): A device object specifying the device to be used (e.g. "cpu" or "cuda").
        txts (list): List of input texts.
        mlmscoreType (str): MLM scoring type ('original' or 'modified')
        max_batch_samp (int): Maximum number of samples to process in a batch.

    Returns:
        mlm_scores_sum (torch.Tensor): Sum of MLM scores over all input text.
        mlm_scores_mean (torch.Tensor): Mean of MLM scores over all input text.
        mlm_scores_each (torch.Tensor): MLM scores for each input text.
    """

    skip_token_ids = [tokenizer.convert_tokens_to_ids(tok_id) for tok_id in list(tokenizer.special_tokens_map.values()) if tok_id not in [tokenizer.mask_token,tokenizer.unk_token]]

    # Tokenize input texts and create masked versions
    inputs_org = tokenizer(txts, padding=True, return_tensors="pt").to(device)
    inputs_masked = {
        'input_ids': [],
        'attention_mask': []
    }
    masked_token_ids = []
    masked_token_posis = []
    
    for inpids, attmask in zip(inputs_org['input_ids'], inputs_org['attention_mask']):
        # for modified MLM scoring Kauf & Ivanova (2023)
        if mlmscoreType == 'modified':
            head_subword_mapping, subword_token_index, head_token_index = construct_head_subword_mapping(tokenizer, device, inpids)
            #print(head_subword_mapping)
        for i in range(len(inpids)):
            masked_inpids = inpids.clone()
            if masked_inpids[i] not in skip_token_ids:
                masked_token_ids.append(inpids[i].clone())
                masked_inpids[i] = tokenizer.mask_token_id
                if mlmscoreType == 'modified' and i in head_token_index:
                    for ix in head_subword_mapping[torch.where(head_token_index==i)[0]][1]:
                        masked_inpids[ix] = tokenizer.mask_token_id
                inputs_masked['input_ids'].append(masked_inpids)
                inputs_masked['attention_mask'].append(attmask)
                masked_token_posis.append(i)
    
    inputs_masked['input_ids'] = torch.vstack(inputs_masked['input_ids'])
    inputs_masked['attention_mask'] = torch.vstack(inputs_masked['attention_mask'])
    masked_token_ids = torch.tensor(masked_token_ids)
    # debug: check masked sentence
    #for inp in inputs_masked['input_ids']:
    #    print(tokenizer.decode(inp))

    # Compute log probabilities from logits for all masked tokens
    nmask_cands = len(masked_token_ids)
    model.eval()
    
    # make minibatch if necessary
    nbatch, modu, n_add = make_minibatch_params(nmask_cands, max_batch_samp)
    mask_token_logps_all = []
    for bi in range(nbatch + n_add):
        sidx = bi * max_batch_samp
        eidx = (bi + 1) * max_batch_samp if bi < nbatch else None

        inputs_masked_tmp = {
            'input_ids': inputs_masked['input_ids'][sidx:eidx],
            'attention_mask': inputs_masked['attention_mask'][sidx:eidx]
        }
        masked_token_posis_tmp = masked_token_posis[sidx:eidx]
        masked_token_ids_tmp = masked_token_ids[sidx:eidx]

        with torch.no_grad():
            logps = torch.nn.functional.log_softmax(model(**inputs_masked_tmp).logits, dim=-1)

        mask_token_logps_each = [logps[ix, mtokposi, mtokid] for ix, (mtokposi, mtokid) in enumerate(zip(masked_token_posis_tmp, masked_token_ids_tmp))]
        mask_token_logps_all += mask_token_logps_each
        del logps, inputs_masked_tmp
        gc.collect(), torch.cuda.empty_cache()    

    
    # Compute MLM scores for each sentence
    lengths = inputs_org['input_ids'].shape[1] - count_special_tokens(inputs_org['input_ids'], skip_token_ids)
    mlm_scores_each = [torch.tensor(mask_token_logps_all[start:end]).to(device) for start, end in zip(torch.cumsum(torch.cat((torch.tensor([0], dtype=torch.int, device=device), lengths)), dim=0)[:-1], torch.cumsum(lengths, dim=0))]
    mlm_scores_sum = torch.tensor([torch.sum(mlmse) for mlmse in mlm_scores_each]).to(device)
    mlm_scores_mean = torch.tensor([torch.mean(mlmse) for mlmse in mlm_scores_each]).to(device)
    
    return mlm_scores_sum, mlm_scores_mean, mlm_scores_each


def compute_mlm_scores_wrapper(model, tokenizer, device, cand_sentences, mlmscoreType='original', nMax_MLMs_cands=5000,mlmsflag=1):
    """
    Wrapper function of MLM scoring computation.

    Args:
        mlmsflag (int): mlmsflag-1 is the number to divide data into multiple batch

    Returns:
        mlm_scores_sum (torch.Tensor): Sum of MLM scores over all input text.
        mlm_scores_mean (torch.Tensor): Mean of MLM scores over all input text.
        mlm_scores_each (torch.Tensor): MLM scores for each input text.
    """

    ncands = len(cand_sentences)
    while True:
        try:
            max_batch_samp_mlm = int(np.ceil(nMax_MLMs_cands/mlmsflag))
            nbatch, modu, n_add = make_minibatch_params(ncands, max_batch_samp_mlm)

            mlm_scores_sum = []
            mlm_scores_mean = []
            mlm_scores_each = []
            for bi in range(nbatch + n_add):
                sidx = bi * max_batch_samp_mlm
                eidx = (bi + 1) * max_batch_samp_mlm if bi < nbatch else None
                mlm_scores_sum0, mlm_scores_mean0, mlm_scores_each0 = compute_mlm_scores(model,tokenizer, device, cand_sentences[sidx:eidx], max_batch_samp=max_batch_samp_mlm, mlmscoreType=mlmscoreType)
                mlm_scores_sum += mlm_scores_sum0
                mlm_scores_mean += mlm_scores_mean0
                mlm_scores_each += mlm_scores_each0
            mlm_scores_sum = torch.tensor(mlm_scores_sum).to(device)
            mlm_scores_mean = torch.tensor(mlm_scores_mean).to(device)
            mlm_scores_each.append(mlm_scores_each)
            break
        except:
            gc.collect(), torch.cuda.empty_cache()
            mlmsflag *= 2
            print('Crash[OOM?]: Restart MLM scoring by dividing data into %d'%(mlmsflag))

    gc.collect(), torch.cuda.empty_cache()    
    return mlm_scores_sum, mlm_scores_mean, mlm_scores_each, mlmsflag



def prepare_masked_candidateIDs(txts, tokenizer, device, nGram4Mask=3, nMaskPerSentence=1, add_insert_mask=True, nMaskCands=5):
    """
    This function prepares all possible masked candidates for a given set of text inputs by replacing n-gram sequences by [MASK]
    and/or by inserting a [MASK] token at various positions in each input text.

    Args:
        txts (List[str]): A list of input texts.
        tokenizer (Tokenizer): An instance of the Tokenizer class from the Hugging Face transformers library.
        device (torch.device): A device object specifying the device to be used (e.g. "cpu" or "cuda").
        nGram4Mask (int, optional): The n-gram size for which the function will replace tokens by [MASK] tokens. Defaults to 3.
        nMaskPerSentence (int, optional): The max number of [MASK] in each sentence. Defaults to 1. Set at most 3 for computational efficiency with the current implementation.
        add_insert_mask (bool, optional): Whether to add a [MASK] token at various positions in the input text. Defaults to True.
        nMaskCands (int, optional): The number of masked candidates to generate per input text. Defaults to 5.
        
    Returns:
        inputs_masked_all (list of lists): List of masked input sentences, where each sentence is a list of token IDs
    """
    
    # step 1. prepare masked sentences as token IDs
    inputs_base = tokenizer(txts, padding=True, return_tensors="pt").to(device)
    skip_token_ids = [tok_id for tok_id in tokenizer.all_special_ids if tok_id not in [tokenizer.mask_token_id,tokenizer.unk_token_id]]
    skip_token_ids_nosep = [tok_id for tok_id in tokenizer.all_special_ids if tok_id not in [tokenizer.mask_token_id,tokenizer.sep_token_id]]

    inputs_masked_all = {
        'input_ids': [],
        'attention_mask': []
    }

    # get sentence length without special tokens
    sentlen = inputs_base['input_ids'].shape[1]-count_special_tokens(inputs_base['input_ids'],skip_token_ids)

    for inpids, attmask, slen in zip(inputs_base['input_ids'], inputs_base['attention_mask'], sentlen):

        inputs_masked = {
            'input_ids': [],
            'attention_mask': []
        }
        nvariation = 1

        # step 1. prepare all possible masked candidates as token IDs
        for ixx in range(nMaskPerSentence):
            if ixx > 0:
                nvariation = len(inputs_masked['input_ids'])
            for ivar in range(nvariation):
                if ixx == 0:
                    masked_inpids = inpids.clone()
                    masked_attmask = attmask.clone()
                else:
                    masked_inpids = inputs_masked['input_ids'][ivar].clone()
                    masked_attmask = inputs_masked['attention_mask'][ivar].clone()

                for i in range(len(masked_inpids)):
                    if masked_inpids[i] not in skip_token_ids:
                        # 1a. replace n-gram sequence by [MASK]
                        for ngrammask in range(1,nGram4Mask+1):
                            if slen < ngrammask:
                                continue
                            masked_inpids2 = torch.cat([masked_inpids[0:i], torch.tensor([tokenizer.mask_token_id]).to(device), masked_inpids[i+ngrammask:]])
                            masked_attmask2 = torch.cat([masked_attmask[0:i], torch.tensor([1]).to(device), masked_attmask[i+ngrammask:]])

                            # add [SEP] token if masked
                            if tokenizer.sep_token_id not in masked_inpids2:
                                if tokenizer.pad_token_id not in masked_inpids2:
                                    masked_inpids2 = torch.cat([masked_inpids2,torch.tensor([tokenizer.sep_token_id]).to(device)])
                                    masked_attmask2 = torch.cat([masked_attmask2,torch.tensor([1]).to(device)])
                                else:
                                    padidx = torch.where(masked_inpids2 == tokenizer.pad_token_id)[0][0]
                                    masked_inpids2[padidx] = tokenizer.sep_token_id
                                    masked_attmask2[padidx] = 1

                            inputs_masked['input_ids'].append(masked_inpids2)
                            inputs_masked['attention_mask'].append(masked_attmask2)

                        # 1b. insert mask
                        if add_insert_mask:
                            if masked_inpids[i] not in skip_token_ids_nosep:
                                masked_inpids2 = torch.cat([masked_inpids[0:i], torch.tensor([tokenizer.mask_token_id]).to(device), masked_inpids[i:]])
                                masked_attmask2 = torch.cat([masked_attmask[0:i], torch.tensor([1]).to(device), masked_attmask[i:]])
                                inputs_masked['input_ids'].append(masked_inpids2)
                                inputs_masked['attention_mask'].append(masked_attmask2)

        # 1c. padding
        maxseqlen = torch.max(torch.tensor([len(inpids) for inpids in inputs_masked['input_ids']]))
        inputs_masked['input_ids'] = torch.vstack([torch.nn.ConstantPad1d((0,maxseqlen-len(i)),tokenizer.pad_token_id)(i) for i in inputs_masked['input_ids']])
        inputs_masked['attention_mask'] = [torch.nn.ConstantPad1d((0,maxseqlen-len(i)),0)(i) for i in inputs_masked['attention_mask']]

        # 1d. get unique
        unique, inverse, counts, index = get_unique(inputs_masked['input_ids'],dim=0)
        inputs_masked['input_ids'] = unique
        inputs_masked['attention_mask'] = torch.vstack([inputs_masked['attention_mask'][i] for i in index])

        # 1e. randomly select masked candidate for efficient search
        ncands_all = len(inputs_masked['input_ids'])
        cand_idx = random.sample(range(ncands_all), k=np.min((ncands_all,nMaskCands)))
        inputs_masked_all['input_ids'].append(torch.vstack([inputs_masked['input_ids'][ci] for ci in cand_idx]))
        inputs_masked_all['attention_mask'].append(torch.vstack([inputs_masked['attention_mask'][ci] for ci in cand_idx]))

    # 1f. padding
    maxseqlen = torch.max(torch.cat([torch.tensor([len(inps) for inps in input_ids]) for input_ids in inputs_masked_all['input_ids']]))
    inputs_masked_all['input_ids'] = torch.vstack([torch.vstack([torch.nn.ConstantPad1d((0,maxseqlen-len(i)),tokenizer.pad_token_id)(i) for i in items]) for items in inputs_masked_all['input_ids']])
    inputs_masked_all['attention_mask'] = torch.vstack([torch.vstack([torch.nn.ConstantPad1d((0,maxseqlen-len(i)),tokenizer.pad_token_id)(i) for i in items]) for items in inputs_masked_all['attention_mask']])

    # 1g. get unique
    unique, inverse, counts, index = get_unique(inputs_masked_all['input_ids'],dim=0)
    inputs_masked_all['input_ids'] = unique
    inputs_masked_all['attention_mask'] = torch.vstack([inputs_masked_all['attention_mask'][i] for i in index])

    return inputs_masked_all



def generate_candidate_sentences_fromIDs(inputs_masked, model, tokenizer, device, mlm_sampling_type='topk', topk=5, skip_token_ids=[],multiMaskType='independent',add_mask_removal=False):
    """
    Generates candidate sentences with masked language modeling.

    Args:
        inputs_masked (list of lists): List of masked input sentences, where each sentence is a list of token IDs
        model (torch.nn.Module): The masked language model to use for inference.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to convert text to input IDs.
        device (str): The device to run the model on (e.g. "cuda:0" for GPU).
        mlm_sampling_type (str): The type of sampling to use for selecting tokens. Can be "topk" or "sampling".
        topk (int): The number of top-scoring tokens to consider for "topk" sampling.
        skip_token_ids (list): A list of token IDs to skip during sampling.
        multiMaskType (str) : The type of filling strategy for multiple masks in each sentence. Can be ["independent" (fast,defualt), ["forward_seq", "backward_seq", "random_seq" (slow, likely to generate words with multi-subwords)].
        add_mask_removal : (bool, optional): Whether to add a candidate that revmove [MASK]. Defaults to False. A similar functionality is complmented by nGram masking.

    Returns:
        A list of candidate sentences with the [MASK] tokens replaced.
    """
    # Convert masked text to input IDs
    # initialize
    cand_sentences = []
    nMaskInSamp = torch.sum(inputs_masked['input_ids'] == tokenizer.mask_token_id,dim=1)
    nMaxMasks = torch.max(nMaskInSamp)
    skip_token_ids_nomask = [tok_id for tok_id in skip_token_ids if tok_id != tokenizer.mask_token_id]

    # fill mask until all are unmasked
    for nmaxitr in range(nMaxMasks):

        inputs_unmasked = {
            'input_ids': [],
            'attention_mask': []
        }

        # Compute logits for all masked tokens
        model.eval()
        with torch.no_grad():
            logits_all = model(**inputs_masked).logits

        # fill mask until all are unmasked
        for logits, inpids, attmask in zip(logits_all, inputs_masked['input_ids'],  inputs_masked['attention_mask']):

            # get nmask info
            tokenIdx = torch.where(inpids == tokenizer.mask_token_id)[0] # lists of token idx

            # get token indices of a single mask token from each sample
            # 'XX_seq' types fill multiple masks sequentially, whereas 'independent' type fills multiple masks independently in parallel
            if multiMaskType == 'forward_seq':
                tokenIdx_tmp = [tokenIdx[0]]
            elif multiMaskType == 'backward_seq':
                tokenIdx_tmp = [tokenIdx[-1]]
            elif multiMaskType == 'random_seq':
                tokenIdx_tmp = random.sample(list(tokenIdx),1)
            elif multiMaskType == 'independent':
                tokenIdx_tmp = random.sample(list(tokenIdx),len(tokenIdx))

            # Aggregate logits for each masked token while avoiding skip tokens from prediction
            for ix, tokidx in enumerate(tokenIdx_tmp):
                logit = logits[tokidx,:].clone()
                # set minimum logit to skip tokens
                for ixx in skip_token_ids:
                    logit[ixx] = torch.finfo().min

                # fill [MASK] tokens by selected candidate tokens
                if mlm_sampling_type == 'topk':
                    k_selected_tokens = torch.topk(logit, topk, dim=0).indices.tolist()
                elif mlm_sampling_type == 'sampling':
                    k_selected_tokens = sample_without_replacement(logit, topk).tolist()

                # set predicted tokens
                for ixx, k_selected_token in enumerate(k_selected_tokens):
                    if ix == 0:
                        unmasked_inpids = inpids.clone()
                        unmasked_inpids[tokidx] = k_selected_token
                        inputs_unmasked['input_ids'].append(unmasked_inpids)
                        inputs_unmasked['attention_mask'].append(attmask.clone())
                    else: # for 'independent' type
                        # update by filling repeatedly
                        inputs_unmasked['input_ids'][ixx] = unmasked_inpids
                        # attention mask needs not edited here.

                    if add_mask_removal:
                        if ix == 0:
                            # Add a [MASK]-removed candidate
                            unmasked_inpids = inpids.clone()
                            inputs_unmasked['input_ids'].append(torch.cat([unmasked_inpids[:tokidx],unmasked_inpids[tokidx+1:],torch.tensor([tokenizer.pad_token_id],device=device)]))
                            inputs_unmasked['attention_mask'].append(torch.cat([attmask[:tokidx],attmask[tokidx+1:],torch.tensor([0],device=device)]))

        # stack all samples
        inputs_unmasked['input_ids'] = torch.vstack(inputs_unmasked['input_ids'])
        inputs_unmasked['attention_mask']= torch.vstack(inputs_unmasked['attention_mask'])


        # separate filled and unfilled samples
        filled_idx = torch.where(torch.sum(inputs_unmasked['input_ids'] == tokenizer.mask_token_id, dim=1) == 0)[0]
        input_ids_filled = [inputs_unmasked['input_ids'][fi] for fi in filled_idx]
        cand_sentences += [tokenizer.decode([inp for inp in inpids if inp not in skip_token_ids]) for inpids in input_ids_filled]

        masked_idx = torch.where(torch.sum(inputs_unmasked['input_ids'] == tokenizer.mask_token_id, dim=1) > 0)[0]
        if len(masked_idx) == 0: # if empty
            break
        inputs_masked['input_ids'] = torch.vstack([inputs_unmasked['input_ids'][mi] for mi in masked_idx])
        inputs_masked['attention_mask'] = torch.vstack([inputs_unmasked['attention_mask'][mi] for mi in masked_idx])

        # use only one candidate for non-first masks to reduce # of candidates (tentative)
        topk = 1 

    # return unique sentence
    return list(set(cand_sentences))


def generate_candidate_sentences_fromIDs_wrapper(inputs_masked, model, tokenizer, device, mlm_sampling_type='topk', topk=5, skip_token_ids=[],multiMaskType='independent', add_mask_removal=False, max_batch_samp=1000):
    
    # generate candidate sentences
    nmask_cands = len(inputs_masked['input_ids'])
    # if ncands is too large, divide data into minibatch
    nbatch, modu, n_add = make_minibatch_params(nmask_cands, max_batch_samp)
    inputs_masked_tmp = {
        'input_ids': [],
        'attention_mask': []
    }

    cand_sentences = []
    for bi in range(nbatch+n_add):
        sidx = bi*max_batch_samp
        eidx = (bi+1)*max_batch_samp if bi < nbatch else None
        inputs_masked_tmp['input_ids'] = inputs_masked['input_ids'][sidx:eidx]
        inputs_masked_tmp['attention_mask'] = inputs_masked['attention_mask'][sidx:eidx]

        cand_sentences_tmp = generate_candidate_sentences_fromIDs(inputs_masked_tmp, model, tokenizer, device, mlm_sampling_type=mlm_sampling_type, topk=topk, skip_token_ids=skip_token_ids, multiMaskType=multiMaskType,add_mask_removal=add_mask_removal)
        cand_sentences += cand_sentences_tmp

    return cand_sentences





def prepare_masked_candidates(txts, tokenizer, device, skip_token_ids, nGram4Mask=3, nMaskPerSentence=1, maxlen=30, add_insert_mask=True, nMaskCands=5, maskingUnitType='word'):
    """
    This function prepares all possible masked candidates for a given set of text inputs by replacing n-gram sequences by [MASK]
    and/or by inserting a [MASK] token at various positions in each input text.

    Args:
        txts (List[str]): A list of input texts.
        tokenizer (Tokenizer): An instance of the Tokenizer class from the Hugging Face transformers library.
        device (torch.device): A device object specifying the device to be used (e.g. "cpu" or "cuda").
        skip_token_ids (List[int]): A list of token IDs to be skipped.
        nGram4Mask (int, optional): The n-gram size for which the function will replace tokens by [MASK] tokens. Defaults to 3.
        nMaskPerSentence (int, optional): The max number of [MASK] in each sentence. Defaults to 1. Set at most 3 for computational efficiency with the current implementation.
        maxlen (int, optional): The maximum length of the input text. Defaults to 30.
        add_insert_mask (bool, optional): Whether to add a [MASK] token at various positions in the input text. Defaults to True.
        nMaskCands (int, optional): The number of masked candidates to generate per input text. Defaults to 5.
        maskingUnitType: Unit of masking (default='word' or 'token')
        
    Returns:
        txts_masked (List[str]): A list of unique masked sentence candidates.
    """
    txts_masked = []    
    for txt in txts:

        # initialize 
        txts_masked_tmp = []

        # 1. prepare all possible masked candidates
        for ixx in range(nMaskPerSentence):
            # [optional]: add additional mask (if nMaskPerSentence > 1)
            if ixx == 0:
                txts_masked_base = [txt]
            else:
                txts_masked_base = set(list(txts_masked_tmp))
            skip_token_ids_tmp = [token_id for token_id in skip_token_ids if token_id != tokenizer.mask_token_id]

            for txt0 in txts_masked_base:
                if maskingUnitType == 'word':
                    inptmp = txt0.split()
                    sentlen = len(inptmp)

                    # 1a. replace n-gram sequence by [MASK]
                    for ngrammask in range(1,nGram4Mask+1):
                        if sentlen < ngrammask:
                            continue
                        maskidx = range(np.min((sentlen,maxlen))-(ngrammask-1))
                        for mi in maskidx:
                            inptmp2 = ' '.join(inptmp[:mi]+[tokenizer.mask_token]+inptmp[mi+ngrammask:])
                            txts_masked_tmp.append(inptmp2)

                    # 1b. insert mask
                    if add_insert_mask:
                        maskidx = range(0,sentlen+1)
                        for mi in maskidx:
                            inp_masked = ' '.join(inptmp[:mi]+[tokenizer.mask_token]+inptmp[mi:])
                            txts_masked_tmp.append(inp_masked)
                    
                elif maskingUnitType == 'token':
                    inputs = tokenizer(txt0, padding=True, return_tensors="pt").to(device) # padding=True is necessary if input length differ across sentences
                    inptmp = torch.tensor([ix for ix in inputs['input_ids'][0] if ix not in skip_token_ids_tmp])
                    sentlen = len(inptmp)

                    # 1a. replace n-gram sequence by [MASK]
                    for ngrammask in range(1,nGram4Mask+1):
                        if sentlen < ngrammask:
                            continue
                        maskidx = range(np.min((sentlen,maxlen))-(ngrammask-1))
                        for mi in maskidx:
                            inptmp2 = torch.hstack((inptmp[:mi],torch.tensor(tokenizer.mask_token_id),inptmp[mi+ngrammask:])).to(torch.int32) # hstack is better than cat as it can include zero-dimensional arguments
                            txts_masked_tmp.append(tokenizer.decode(inptmp2))
                            #txts_masked_tmp.append(' '.join([tokenizer.decode(int(ix)) for ix in inptmp2]))

                    # 1b. insert mask
                    if add_insert_mask:
                        maskidx = range(0,sentlen+1)
                        for mi in maskidx:
                            inp_masked = torch.hstack((inptmp[:mi],torch.tensor(tokenizer.mask_token_id),inptmp[mi:])).to(torch.int32) # hstack is better than cat as it can include zero-dimensional arguments
                            txts_masked_tmp.append(tokenizer.decode(inp_masked))
                            #txts_masked_tmp.append(' '.join([tokenizer.decode(int(ix)) for ix in inp_masked]))

        # 2. randomly select masked candidate for efficient search
        cand_idx = random.sample(range(len(txts_masked_tmp)), k=np.min((len(txts_masked_tmp),nMaskCands)))
        txts_masked += [txts_masked_tmp[ci] for ci in cand_idx]

    # get unique masked sentence
    txts_masked = list(set(txts_masked))

    return txts_masked


def generate_candidate_sentences(txts_masked, model, tokenizer, device, mlm_sampling_type='topk', topk=5, skip_token_ids=[],multiMaskType='independent'):
    """
    Generates candidate sentences with masked language modeling.

    Args:
        txts_masked (list): A list of strings with the [MASK] tokens to be replaced.
        model (torch.nn.Module): The masked language model to use for inference.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to convert text to input IDs.
        device (str): The device to run the model on (e.g. "cuda:0" for GPU).
        mlm_sampling_type (str): The type of sampling to use for selecting tokens. Can be "topk" or "sampling".
        topk (int): The number of top-scoring tokens to consider for "topk" sampling.
        skip_token_ids (list): A list of token IDs to skip during sampling.
        multiMaskType (str) : The type of filling strategy for multiple masks in each sentence. Can be ["independent" (fast,defualt), ["forward_seq", "backward_seq", "random_seq" (slow, likely to generate words with multi-subwords)].

    Returns:
        A list of candidate sentences with the [MASK] tokens replaced.
    """
    # Convert masked text to input IDs

    # initialize
    cand_sentences = []
    nMaskInSamp = torch.tensor([tm.count(tokenizer.mask_token) for tm in txts_masked])
    nMaxMasks = torch.max(nMaskInSamp)
    skip_token_ids_nomask = [tok_id for tok_id in skip_token_ids if tok_id != tokenizer.mask_token_id]

    # fill mask until all are unmasked
    for nmaxitr in range(nMaxMasks):
        inputs = tokenizer(txts_masked, padding=True, return_tensors="pt").to(device)

        # get nmask info
        mask_token_index = torch.where(inputs['input_ids'] == tokenizer.mask_token_id) # lists of sample and token idx
        ncand_samps = len(txts_masked)
        sampIdx = mask_token_index[0]
        tokenIdx = mask_token_index[1]

        # Compute logits for all masked tokens
        model.eval()
        with torch.no_grad():
            logits = model(**inputs).logits

        # get sample&token indices of a single mask token from each sample
        # 'XX_seq' types fill multiple masks sequentially, whereas 'independent' type fills multiple masks independently in parallel
        sampIdx_tmp = torch.hstack([sampIdx[torch.where(sampIdx == ix)][0] for ix in range(ncand_samps)])
        if multiMaskType == 'forward_seq':
            tokenIdx_tmp = torch.hstack([tokenIdx[torch.where(sampIdx == ix)][0] for ix in range(ncand_samps)])
        elif multiMaskType == 'backward_seq':
            tokenIdx_tmp = torch.hstack([tokenIdx[torch.where(sampIdx == ix)][-1] for ix in range(ncand_samps)])
        elif multiMaskType == 'random_seq':
            tokenIdx_tmp = torch.hstack([torch.hstack(random.sample(list(tokenIdx[torch.where(sampIdx == ix)]),len(torch.where(sampIdx == ix)[0])))[0] for ix in range(ncand_samps)])
        elif multiMaskType == 'independent':
            sampIdx_tmp = sampIdx.clone().detach() # rewrite sampIdx for independent masking
            tokenIdx_tmp = tokenIdx.clone().detach()

        # Aggregate logits for each masked token while avoiding skip tokens from prediction
        mask_token_logits_all = []
        mask_token_posi_all = []
        for ix in range(ncand_samps):
            logit = logits[ix,:,:]

            # identify samples and tokens corresponding to [MASK]
            sampidx = torch.where(sampIdx_tmp == ix)
            tokenidx = tokenIdx_tmp[sampidx]

            # set minimum logit to skip tokens
            for tix in tokenidx:
                for ixx in skip_token_ids:
                    logit[tix, ixx] = torch.finfo().min #torch.min(logit[tix, :])

            # keep tokenidx and logit
            mask_token_posi_all.append(tokenidx)
            mask_token_logits_all.append([logit[tix, :] for tix in tokenidx])


        # Construct candidate sentences with a specifid selection method (e.g.,top-k, categorical sampling)
        cand_sentences_tmp = []
        # Create [MASK]-filled sentences
        for txt_, mask_token_logits, mask_token_posi, inps in zip(txts_masked, mask_token_logits_all, mask_token_posi_all, inputs['input_ids']):
            # fill all [MASK] tokens by selected candidate tokens
            for ix, (mtl, mtp) in enumerate(zip(mask_token_logits, mask_token_posi)):

                if mlm_sampling_type == 'topk':
                    k_selected_tokens = torch.topk(mtl, topk, dim=0).indices.tolist()
                elif mlm_sampling_type == 'sampling':
                    k_selected_tokens = sample_without_replacement(mtl, topk).tolist()

                # filling
                if ix == 0:
                    # fill first mask
                    inps_filled = inps.clone().detach()
                    cand_sentences_ids = []
                    for token in k_selected_tokens:
                        inps_filled[mtp] = token
                        cand_sentences_ids.append(inps_filled.clone().detach())
                else: # for 'independent' filling
                    new_cand_sentences_ids = []
                    for ixx,token in enumerate(k_selected_tokens):
                        inps_filled = cand_sentences_ids[ixx].clone().detach()
                        inps_filled[mtp] = token
                        new_cand_sentences_ids.append(inps_filled.clone().detach())
                    cand_sentences_ids = new_cand_sentences_ids

            # decode to string
            for csi in cand_sentences_ids:
                cand_sentences_tmp.append(tokenizer.decode(csi[~torch.stack([csi == sti for sti in skip_token_ids_nomask]).any(dim=0)]))

            # Add a [MASK]-removed candidate
            cand_sentences_tmp.append(txt_.replace(tokenizer.mask_token+' ', '').replace(' '+tokenizer.mask_token, ''))
            
        # aggregate fully unmasked samples
        cand_sentences_tmp = list(set(cand_sentences_tmp))
        
        nMaskInSamp = torch.tensor([cst.count(tokenizer.mask_token) for cst in cand_sentences_tmp])
        txts_masked = [cand_sentences_tmp[ixx] for ixx in torch.where(nMaskInSamp > 0)[0]]
        txts_unmasked = [cand_sentences_tmp[ixx] for ixx in torch.where(nMaskInSamp == 0)[0]]
        cand_sentences += txts_unmasked

        # update params
        # use only one candidate for non-first masks to reduce # of candidates (tentative)
        topk = 1 
        if not txts_masked: # if empty
            break

    # return unique sentence
    return list(set(cand_sentences))


def generate_candidate_sentences_wrapper(txts_masked, model, tokenizer, device, mlm_sampling_type='topk', topk=5, skip_token_ids=[],multiMaskType='independent', max_batch_samp=1000):
    
    # generate candidate sentences
    nmask_cands = len(txts_masked)
    # if ncands is too large, divide data into minibatch
    nbatch, modu, n_add = make_minibatch_params(nmask_cands, max_batch_samp)

    cand_sentences = []
    for bi in range(nbatch+n_add):
        sidx = bi*max_batch_samp
        eidx = (bi+1)*max_batch_samp if bi < nbatch else None
        txts_masked_tmp = txts_masked[sidx:eidx]

        cand_sentences_tmp = generate_candidate_sentences(txts_masked_tmp, model, tokenizer, device, mlm_sampling_type=mlm_sampling_type, topk=topk, skip_token_ids=skip_token_ids_mlm, multiMaskType=multiMaskType)
        cand_sentences += cand_sentences_tmp

    return cand_sentences

def compute_sentence_feature_patterns(model_lm, tokenizer_lm, input_ids, attention_mask, skip_token_ids=[], feat_mu_all=[], feat_sd_all=[], do_norm=False, device=torch.device('cpu'), layerIdx=[]):
    """
    Compute sentence feature patterns for individual sentences by averaging word features produced from a pretrained language model using the given input sentences and attention masks.
    
    Args:
        - model_lm (torch.nn.Module): Pretrained language model
        - tokenizer_lm (transformers.tokenization_utils_base.PreTrainedTokenizerBase): Tokenizer for the language model
        - input_ids (list of lists): List of input sentences, where each sentence is a list of token IDs
        - attention_mask (list of lists): List of attention masks, where each mask is a list of 1s and 0s indicating which tokens to attend to
        - skip_token_ids (list, optional): List of token IDs to skip when computing sentence features (default: [])
        - feat_mu_all (list, optional): List of means for each layer of the language model, used for normalization (default: [])
        - feat_sd_all (list, optional): List of standard deviations for each layer of the language model, used for normalization (default: [])
        - do_norm (bool, optional): Whether to normalize the sentence features (default: False)
        - device (torch.device, optional): Device to run the computation on (default: 'cpu')
        - layerIdx (list, optional): Indices of layers used for feature computation.
    
    Returns:
        - feats_all (list of lists): List of sentence features, where each feature is a list of layer-wise means of token embeddings
    
    """
    model_lm.eval()
    with torch.no_grad():
        outputs_lm = model_lm(input_ids=input_ids,attention_mask=attention_mask,output_hidden_states=True).hidden_states
    
    # construct sentence feature patterns for individual sentences by averaging
    feats_all = []
    for oi in range(len(input_ids)):
        ntokens = len(input_ids[oi])
        tokenids = input_ids[oi]
        tokens = tokenizer_lm.convert_ids_to_tokens(input_ids[oi])

        feats = []
        if not layerIdx:
            layerIdx = range(len(outputs_lm)-1)
        for litr in layerIdx:
            tokfeats = []
            for itok,tokenid in enumerate(tokenids):
                if np.all([tokenizer_lm.convert_tokens_to_ids(tokens[itok]) != tok for tok in skip_token_ids]): # contain if not skip tokens
                    tokfeat = outputs_lm[litr+1][oi][itok]
                    tokfeats.append(tokfeat)

            if not tokfeats:
                tokfeats.append(torch.randn(outputs_lm[litr+1][oi][0].shape).to(device))

            tokfeats = torch.vstack(tokfeats)
            tokmean = torch.mean(tokfeats,axis=0)
            if do_norm:
                feats.append((tokmean-feat_mu_all[litr])/feat_sd_all[litr])
            else:
                feats.append(tokmean)

        # aggreagete all candidates
        feats_all.append(feats)

    return feats_all


def compute_sentence_feature_patterns_wrapper(cand_sentences, model_lm, tokenizer_lm, skip_token_ids=[], feat_mu_all=[], feat_sd_all=[], do_norm=False, device=torch.device('cpu'), layerIdx=[], max_batch_samp=1000, divflag=1, divMax=1000):

    # input construction might be better to be done outside of the loop for fixed padded size for all samples.
    inputs = tokenizer_lm(cand_sentences, padding=True, return_tensors="pt").to(device)
    nfeat_cands = len(cand_sentences)

    # if ncands is too large, divide data into minibatch    
    while True:
        try:
            nbatch, modu, n_add = make_minibatch_params(nfeat_cands, int(np.ceil(max_batch_samp/divflag)))
            feats_all = []
            for bi in range(nbatch+n_add):
                sidx = bi*max_batch_samp
                eidx = (bi+1)*max_batch_samp if bi < nbatch else None        
                inputs_tmp = inputs['input_ids'][sidx:eidx]
                attention_mask_tmp = inputs['attention_mask'][sidx:eidx]

                feats_tmp = compute_sentence_feature_patterns(model_lm.to(device), tokenizer_lm, input_ids=inputs_tmp,attention_mask=attention_mask_tmp, skip_token_ids=skip_token_ids, do_norm=do_norm, feat_mu_all=feat_mu_all, feat_sd_all=feat_sd_all, device=device, layerIdx=layerIdx)
                feats_all += feats_tmp 
            break
        except:
            del feats_all 
            gc.collect(), torch.cuda.empty_cache()
            divflag *= 2
            print('Crash[OOM?]: Restart feature computation by dividing data into %d'%(divflag))
            if divflag >= divMax:
                raise Exception("Failed to finish computation. Stop this process.")
           
    return feats_all, inputs


import os
import transformers as trf
import sys
import subprocess
import glob

# function
def getDN(path):
    files = sorted(glob.glob(path)) # ????????????????????????????????????
    return files

# get file names
def getFN(path):
    # files=subprocess.run('ls ' + path + ' | xargs -n 1 basename').split("\n")
    files = sorted([filename.split('/')[-1] for filename in glob.glob(path)]) # 20240222
    return files


# random integers generator
def randsample(min, max, cnt, sortflag=False, revflag=False):
    list = []
    i = 0
    while cnt != i:
        r = random.randint(min, max)
        try:
            list.index(r)
        except ValueError:
            list.append(r)
            i = i + 1
    if (sortflag): list.sort(reverse=revflag)
    return list

# check and set directory
def setdir(path):
    if os.path.exists(path)==False:
        try:
            os.makedirs(path)
            try:
                os.chmod(path,0o777)
                #os.chmod(path,0o755)
            except:
                print('Cannot change the permission of the directory')
        except:
            print('Crash: failed to make directories or it already exists.')
    return path

# fix seed
def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # Tensorflow
    #tf.random.set_seed(seed)


def load_mlm_model(MLMmodeldir, MLMType, proxies, device):
    """
    Function to load a pre-trained masked language model from Hugging Face.

    Args:
        MLMmodeldir (str): Directory path for the MLM model.
        MLMType (str): Type of the MLM model.
        proxies (dict): Proxy configuration, if required.
        device: Device to load the model on.

    Returns:
        tokenizer: Loaded tokenizer for the MLM model.
        model: Loaded MLM model.

    """

    # load pre-trained masked language model
    print('Loading mlm model...')
    model_path = setdir("%s/%s/model_mlm/" % (MLMmodeldir, MLMType))
    tokenizer_path = setdir("%s/%s/tokenizer_mlm/" % (MLMmodeldir, MLMType))
    model_file_path = '%spytorch_model.bin' % (model_path)
    
    if MLMType.startswith('bert'):
        tokenizer_class = trf.BertTokenizerFast
        model_class = trf.BertForMaskedLM
        config_class = trf.BertConfig
        model_load_path = MLMType
    elif MLMType.startswith('roberta'):
        tokenizer_class = trf.RobertaTokenizerFast
        model_class = trf.RobertaForMaskedLM
        config_class = trf.RobertaConfig
        model_load_path = MLMType
    elif MLMType.startswith('albert'):
        tokenizer_class = trf.AlbertTokenizerFast
        model_class = trf.AlbertForMaskedLM
        config_class = trf.AlbertConfig
        model_load_path = MLMType
    elif MLMType.startswith('deberta'):
        tokenizer_class = trf.AutoTokenizer
        model_class = trf.DebertaForMaskedLM
        config_class = trf.DebertaConfig
        model_load_path = 'lsanochkin/' + MLMType
    else:
        raise ValueError("Unsupported model type: {}".format(model_type))
        
    # Check if model and tokenizer exist, else download and save
    if not os.path.exists(model_file_path):
        print('Downloading and saving model and tokenizer...')
        tokenizer = tokenizer_class.from_pretrained(model_load_path.replace('_untrained',''))
        model = model_class.from_pretrained(model_load_path.replace('_untrained','')).to(device)
        tokenizer.save_pretrained(tokenizer_path)
        model.save_pretrained(model_path)
    else:
        print('Loading model and tokenizer from cache...')
        tokenizer = tokenizer_class.from_pretrained(tokenizer_path)
        model = model_class.from_pretrained(model_path).to(device)

    if MLMType.endswith('untrained'):
        print('untrained')
        SEED = 42
        fix_seed(SEED)
        model = model_class(config=config_class.from_pretrained(model_path.replace('_untrained',''))).to(device)

    print('Load {} model done'.format(MLMType))
    return tokenizer, model


def load_lm_model(LMmodeldir, LMType, proxies, device, initialize=False):
    """
    Function to load a large language model from Hugging Face.

    Args:
        LMmodeldir (str): Directory path for the LM model.
        LMType (str): Type of the LM model.
        proxies (dict): Proxy configuration, if required.
        device: Device to load the model on.
        initialize: if 1, initialize model weight with seed=42

    Returns:
        tokenizer: Loaded tokenizer for the LM model.
        model: Loaded LM model.
        nlayers: Number of hidden layers in the model.

    """

    # load feature computation model
    print('Loading lm model...')
    model_path = setdir("%s/%s/model/" % (LMmodeldir, LMType))
    tokenizer_path = setdir("%s/%s/tokenizer/" % (LMmodeldir, LMType))
    model_file_path = '%spytorch_model.bin' % (model_path)

    if LMType.startswith('bert'):
        tokenizer_class = trf.BertTokenizerFast
        model_class = trf.BertModel
        config_class = trf.BertConfig
        model_load_path = LMType
    elif LMType.startswith('roberta'):
        tokenizer_class = trf.RobertaTokenizerFast
        model_class = trf.RobertaModel
        config_class = trf.RobertaConfig
        model_load_path = LMType
    elif LMType.startswith('albert'):
        tokenizer_class = trf.AlbertTokenizerFast
        model_class = trf.AlbertModel
        config_class = trf.AlbertConfig
        model_load_path = LMType
    elif LMType.startswith(('gpt2','distilgpt2')):
        tokenizer_class = trf.GPT2TokenizerFast
        model_class = trf.GPT2Model
        config_class = trf.GPT2Config
        model_load_path = LMType
    elif LMType.startswith('openai-gpt'):
        tokenizer_class = trf.OpenAIGPTTokenizerFast
        model_class = trf.OpenAIGPTModel
        config_class = trf.OpenAIGPTConfig
        model_load_path = LMType
    elif LMType.startswith('gpt-neo-1p3b'):
        tokenizer_class = trf.GPT2TokenizerFast
        model_class = trf.GPTNeoModel
        config_class = trf.GPTNeoConfig
        model_load_path = 'EleutherAI/gpt-neo-1.3B'
    elif LMType.startswith('gpt-neo-2p7b'):
        tokenizer_class = trf.GPT2TokenizerFast
        model_class = trf.GPTNeoModel
        config_class = trf.GPTNeoConfig
        model_load_path = 'EleutherAI/gpt-neo-2.7B'
    elif LMType.startswith('sgpt'):
        tokenizer_class = trf.AutoTokenizer
        model_class = trf.AutoModel
        config_class = trf.AutoConfig
        model_load_path = "Muennighoff/SGPT-125M-weightedmean-nli-bitfit"
    elif LMType.startswith('clip'):
        tokenizer_class = trf.CLIPTokenizerFast
        model_class = trf.CLIPTextModel
        config_class = trf.CLIPTextConfig
        model_load_path = 'openai/clip-vit-base-patch32'
    elif LMType.startswith('xclip'):
        tokenizer_class = trf.AutoTokenizer
        model_class = trf.XCLIPTextModel
        config_class = trf.XCLIPTextConfig
        model_load_path = 'microsoft/xclip-base-patch32'
    elif LMType.startswith('xlnet'):
        tokenizer_class = trf.XLNetTokenizerFast
        model_class = trf.XLNetModel
        config_class = trf.XLNetConfig
        model_load_path = LMType
    elif LMType.startswith('distilbert'):
        tokenizer_class = trf.DistilBertTokenizerFast
        model_class = trf.DistilBertModel
        config_class = trf.DistilBertConfig
        model_load_path = LMType
    elif LMType.startswith('t5'):
        tokenizer_class = trf.T5TokenizerFast
        model_class = trf.T5EncoderModel
        config_class = trf.T5Config
        model_load_path = LMType
    elif LMType.startswith('bart'):
        tokenizer_class = trf.BartTokenizerFast
        model_class = trf.BartForCausalLM
        config_class = trf.BartConfig
        model_load_path = LMType
    elif LMType.startswith('ctrl'):
        tokenizer_class = trf.CTRLTokenizer
        model_class = trf.CTRLModel
        config_class = trf.CTRLConfig
        model_load_path = LMType
    elif LMType.startswith(('xlm-mlm','xlm-clm')):
        tokenizer_class = trf.XLMTokenizer
        model_class = trf.XLMModel
        config_class = trf.XLMConfig
        model_load_path = LMType
    elif LMType.startswith('xlm-roberta'):
        tokenizer_class = trf.XLMRobertaTokenizerFast
        model_class = trf.XLMRobertaModel
        config_class = trf.XLMRobertaConfig
        model_load_path = LMType
    elif LMType.startswith('electra'):
        tokenizer_class = trf.ElectraTokenizerFast
        model_class = trf.ElectraModel
        config_class = trf.ElectraConfig
        model_load_path = LMType
    elif LMType.startswith('deberta'):
        if LMType.find('v2') >= 0:
            tokenizer_class = trf.DebertaV2TokenizerFast
            model_class = trf.DebertaV2Model
            config_class = trf.DebertaV2Config
        elif LMType.find('v3') >= 0:
            tokenizer_class = trf.AutoTokenizer
            model_class = trf.AutoModel
            config_class = trf.AutoConfig
        else:
            tokenizer_class = trf.DebertaTokenizerFast
            model_class = trf.DebertaModel
            config_class = trf.DebertaConfig
        model_load_path = 'microsoft/' + LMType
    else:
        raise ValueError("Unsupported model type: {}".format(model_type))
        
    # Check if model and tokenizer exist, else download and save
    if not os.path.exists(model_file_path):
        print('Downloading and saving model and tokenizer...')
        tokenizer = tokenizer_class.from_pretrained(model_load_path.replace('_untrained',''), proxies=proxies)
        model = model_class.from_pretrained(model_load_path.replace('_untrained',''), proxies=proxies).to(device)
        tokenizer.save_pretrained(tokenizer_path)
        model.save_pretrained(model_path)
    else:
        print('Loading model and tokenizer from cache...')
        tokenizer = tokenizer_class.from_pretrained(tokenizer_path)
        model = model_class.from_pretrained(model_path).to(device)

    if LMType.startswith(('gpt', 'distilgpt2', 'sgpt', 'openai-gpt','xlnet','ctrl','electra')):
        tokenizer.pad_token = tokenizer.unk_token

    if initialize:
        SEED = 42
        fix_seed(SEED)
        model = model_class(config=config_class.from_pretrained(model_path.replace('_untrained',''))).to(device)


    nlayers = model.config.num_hidden_layers
    print('Load {} model done'.format(LMType))
    return tokenizer, model, nlayers




def set_skip_token_ids(tokenizer, speficied_skip_tokens=[], include_special_token=True):
    """
    Returns a list of token IDs to be skipped during tokenization using the given tokenizer.

    Parameters:
        tokenizer (PreTrainedTokenizer): The tokenizer to use for converting tokens to IDs.
        speficied_skip_tokens (list, optional): List of specific tokens to skip (by default, empty list).
        include_special_token (bool, optional): Whether to include special tokens in the skip list (by default, True).

    Returns:
        list: A list of unique token IDs to be skipped during tokenization.
    """

    # Convert specified skip tokens to their respective token IDs
    skip_token_ids = [tokenizer.convert_tokens_to_ids(skip_token) for skip_token in speficied_skip_tokens]

    if include_special_token:
        # Include token IDs of special tokens in the skip list
        special_tokens = flatten_nested_list(list(tokenizer.special_tokens_map.values()))
        for special_token in special_tokens:
            skip_token_ids.append(tokenizer.convert_tokens_to_ids(special_token))

    # Return the list of unique token IDs to be skipped
    return list(set(skip_token_ids))






import time
import torch
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from nltk import bleu_score
from nltk import meteor_score
from nltk import word_tokenize
from bert_score import BERTScorer

def evaluate_nlp_metric_wrapper(tokenizer,caps_each, unique_bests, caps_others, eval_metrics, uni2org_idx, lastIdx, nCapEach, nVidFalse, nUniBests, scorer_base, device, verbose=True):
    """
    Compute various NLP evaluation metrics on a given set of captions.

    Args:
        tokenizer (Tokenizer): Tokenizer for encoding and decoding captions.
        caps_each (list): List of reference captions for each video.
        unique_bests (list): List of unique generated captions for each video.
        caps_others (list): List of referece captions from other models for each video.
        eval_metrics (list): List of evaluation metric names (e.g., 'BLEU', 'METEOR', 'ROUGE-L', 'CIDEr', 'BERTscore').
        uni2org_idx (list): Mapping of unique caption indices to original caption indices.
        lastIdx (int): Index of the last caption for evaluation.
        nCapEach (int): Number of captions for each video.
        nVidFalse (int): Number of false references per video.
        nUniBests (int): Number of unique best captions.
        scorer_base (Model): BertScore model
        device (str): Device for computation.

    Returns:
        dict: A dictionary containing evaluation scores for each metric.
    """
    
    # initialize parameters
    do_nltkflag = 1
    SCORES = {}
    start = time.time()
    
    # compute metrics
    for eval_metric in eval_metrics:
        SCORES[eval_metric] = {}

        # preparations
        if eval_metric in ['BLEU','METEOR'] and do_nltkflag:
            print('Processing nltk tokenization...') if verbose else None
            # prepare computing BLEU and METEOR using nltk
            # pass encode&decode to make case consistent.
            tokenized_refs = [word_tokenize(tokenizer.decode(tokenizer.encode(c, add_special_tokens=False))) for c in caps_each]
            tokenized_bests = [word_tokenize(tokenizer.decode(tokenizer.encode(c, add_special_tokens=False))) for c in unique_bests]
            tokenized_others = [word_tokenize(tokenizer.decode(tokenizer.encode(c, add_special_tokens=False))) for c in caps_others]
            tokenized_others = [tokenized_others[i:i + nCapEach] for i in range(0, len(tokenized_others), nCapEach)]
            do_nltkflag = 0
            
        elif eval_metric in ['ROUGE-L']:
            print('Preprocessing for ROUGE-L computation...') if verbose else None
            # construct inputs of rouge
            # Preprocess reference captions
            preprocessed_refs = [tokenizer.decode(tokenizer.encode(c, add_special_tokens=False)) for c in caps_each]*nUniBests
            index_refs = flatten_list([[i]*nCapEach for i in range(0,nUniBests)])
            ref_set = convert_index_and_captions_to_dict(index_refs, preprocessed_refs)
            # Preprocess best captions
            preprocessed_bests = [tokenizer.decode(tokenizer.encode(c, add_special_tokens=False)) for c in unique_bests]
            index_bests = range(0,nUniBests)
            best_set = convert_index_and_captions_to_dict(index_bests, preprocessed_bests)
            # Preprocess other captions
            preprocessed_others = [tokenizer.decode(tokenizer.encode(c, add_special_tokens=False)) for c in caps_others]
            preprocessed_others = [preprocessed_others[i:i + nCapEach] for i in range(0, len(preprocessed_others), nCapEach)]
            index_others = flatten_list([[i]*nCapEach for i in range(0,nUniBests)])
            
        elif eval_metric in ['CIDEr']:
            print('Preprocessing for CIDEr computation...') if verbose else None
            # construct inputs of cider
            # Preprocess reference and others captions
            preprocessed_refs = [tokenizer.decode(tokenizer.encode(c, add_special_tokens=False)) for c in caps_each]*nUniBests
            preprocessed_others_tmp = [tokenizer.decode(tokenizer.encode(c, add_special_tokens=False)) for c in caps_others]
            preprocessed_others_tmp2 = [preprocessed_others_tmp[i:i + nCapEach] for i in range(0, len(preprocessed_others_tmp), nCapEach)]
            preprocessed_others = [preprocessed_other*nUniBests for preprocessed_other in preprocessed_others_tmp2]
            preprocessed_true_falses = flatten_list([preprocessed_refs]+preprocessed_others)
            index_true_false_refs = flatten_list([[i]*nCapEach for i in range(0,(nVidFalse+1)*nUniBests)])
            ref_true_false_set = convert_index_and_captions_to_dict(index_true_false_refs, preprocessed_true_falses)
            # Preprocess best captions
            preprocessed_bests = [tokenizer.decode(tokenizer.encode(c, add_special_tokens=False)) for c in unique_bests]*(nVidFalse+1)
            index_bests = range(0,nUniBests*(nVidFalse+1))
            best_set = convert_index_and_captions_to_dict(index_bests, preprocessed_bests)
            
        elif eval_metric in ['BERTscore']:
            # prepare inputs
            bs_p_inputs_true_ref = flatten_list([[tokenizer.decode(tokenizer.encode(c, add_special_tokens=False))]*nCapEach for c in unique_bests])
            bs_t_inputs_true_ref = [tokenizer.decode(tokenizer.encode(c, add_special_tokens=False)) for c in caps_each]*nUniBests
            bs_p_inputs_false_ref = flatten_list([[tokenizer.decode(tokenizer.encode(c, add_special_tokens=False))]*len(caps_others) for c in unique_bests])
            bs_t_inputs_false_ref = caps_others*nUniBests
            bertscore_types = ['F1','P','R']
            
        # Processing
        print('%s:'%(eval_metric)) if verbose else None
        if eval_metric in ['BLEU']:
            smoothfunc = bleu_score.SmoothingFunction().method2
            # compute BLEU scores
            scores_true_refs = torch.tensor([bleu_score.sentence_bleu(tokenized_refs, c, smoothing_function=smoothfunc) for c in tokenized_bests]).to(device)
            print('[true ref] :score = %.4f [t=%.5f]'%(scores_true_refs[lastIdx], time.time() - start)) if verbose else None
            scores_false_refs = torch.tensor([[bleu_score.sentence_bleu(tokenized_other, c, smoothing_function=smoothfunc) for tokenized_other in tokenized_others] for c in tokenized_bests]).to(device)
            idens_scores = torch.tensor([compute_identifications_scores(scores_true_refs[nitr], scores_false_refs[nitr,:], summaryType='max', device=device) for nitr in range(nUniBests)]).to(device)
            print('[false ref]:score = %.4f [t=%.5f]' %(torch.mean(scores_false_refs[lastIdx,:]),time.time() - start)) if verbose else None
            print('[iden acc] :cr = %.4f%% [t=%.5f]' % (idens_scores[lastIdx] * 100, time.time() - start)) if verbose else None

        elif eval_metric in ['METEOR']:
            # compute METEOR scores
            scores_true_refs = torch.tensor([meteor_score.meteor_score(tokenized_refs, c) for c in tokenized_bests]).to(device)
            print('[true ref] :score = %.4f [t=%.5f]'%(scores_true_refs[lastIdx], time.time() - start)) if verbose else None
            scores_false_refs = torch.tensor([[meteor_score.meteor_score(tokenized_other, c) for tokenized_other in tokenized_others] for c in tokenized_bests]).to(device)
            idens_scores = torch.tensor([compute_identifications_scores(scores_true_refs[nitr], scores_false_refs[nitr,:], summaryType='max', device=device) for nitr in range(nUniBests)]).to(device)
            print('[false ref]:score = %.4f [t=%.5f]' %(torch.mean(scores_false_refs[lastIdx,:]),time.time() - start)) if verbose else None
            print('[iden acc] :cr = %.4f%% [t=%.5f]' % (idens_scores[lastIdx] * 100, time.time() - start)) if verbose else None

        elif eval_metric in ['ROUGE-L']:
            # compute ROUGE-L scores
            scores_true_refs = torch.tensor(Rouge().compute_score(ref_set,best_set)[1]).to(device)
            print('[true ref] :score = %.4f [t=%.5f]'%(scores_true_refs[lastIdx], time.time() - start)) if verbose else None
            scores_false_refs = torch.tensor(np.array([Rouge().compute_score(convert_index_and_captions_to_dict(index_others, preprocessed_other*nUniBests),best_set)[1] for preprocessed_other in preprocessed_others])).T.to(device)
            idens_scores = torch.tensor([compute_identifications_scores(scores_true_refs[nitr], scores_false_refs[nitr,:], summaryType='max', device=device) for nitr in range(nUniBests)]).to(device)
            print('[false ref]:score = %.4f [t=%.5f]' %(torch.mean(scores_false_refs[lastIdx,:]),time.time() - start)) if verbose else None
            print('[iden acc] :cr = %.4f%% [t=%.5f]' % (idens_scores[lastIdx] * 100, time.time() - start)) if verbose else None

        elif eval_metric in ['CIDEr']:
            # compute CIDEr scores
            scores_true_false_refs = Cider().compute_score(ref_true_false_set,best_set)[1]
            scores_true_refs = torch.tensor(scores_true_false_refs.reshape(nVidFalse+1,nUniBests).T[:,0]).to(device)
            scores_false_refs = torch.tensor(scores_true_false_refs.reshape(nVidFalse+1,nUniBests).T[:,1:]).to(device)
            print('[true ref] :score = %.4f [t=%.5f]'%(scores_true_refs[lastIdx], time.time() - start)) if verbose else None
            idens_scores = torch.tensor([compute_identifications_scores(scores_true_refs[nitr], scores_false_refs[nitr,:], summaryType='max', device=device) for nitr in range(nUniBests)]).to(device)
            print('[false ref]:score = %.4f [t=%.5f]' %(torch.mean(scores_false_refs[lastIdx,:]),time.time() - start)) if verbose else None
            print('[iden acc] :cr = %.4f%% [t=%.5f]' % (idens_scores[lastIdx] * 100, time.time() - start)) if verbose else None

        elif eval_metric in ['BERTscore']:
            # compute bert_scores
            P, R, F1 = scorer_base.score(bs_p_inputs_true_ref,bs_t_inputs_true_ref)
            P, R, F1 = P.reshape((nUniBests,nCapEach)).to(device), R.reshape((nUniBests,nCapEach)).to(device), F1.reshape((nUniBests,nCapEach)).to(device)
            print('F1[true ref] ::max:score = %.4f; mean:score = %.4f [t=%.5f]'%(torch.max(F1[lastIdx]),torch.mean(F1[lastIdx]), time.time() - start)) if verbose else None
            print(' P[true ref] ::max:score = %.4f; mean:score = %.4f [t=%.5f]'%(torch.max(P[lastIdx]),torch.mean(P[lastIdx]), time.time() - start)) if verbose else None
            print(' R[true ref] ::max:score = %.4f; mean:score = %.4f [t=%.5f]'%(torch.max(R[lastIdx]),torch.mean(R[lastIdx]), time.time() - start)) if verbose else None
            # compute scores for other videos
            P_false_refs, R_false_refs, F1_false_refs = scorer_base.score(bs_p_inputs_false_ref,bs_t_inputs_false_ref)
            P_false_refs, R_false_refs, F1_false_refs = P_false_refs.reshape((nUniBests,len(caps_others))).to(device), R_false_refs.reshape((nUniBests,len(caps_others))).to(device), F1_false_refs.reshape((nUniBests,len(caps_others))).to(device)

            # summarize false reference comparisons for BERT scores
            for btype in bertscore_types:
                if btype == 'F1':
                    scores_true_refs = F1
                    scores_false_refs = F1_false_refs
                elif btype == 'P':
                    scores_true_refs = P
                    scores_false_refs = P_false_refs
                elif btype == 'R':
                    scores_true_refs = R
                    scores_false_refs = R_false_refs

                scores_false_refs_max = torch.vstack([step_summarize_scores(scores_false_refs[nitr,:],nCapEach, summaryType='max', device=device) for nitr in range(nUniBests)]).to(device)
                idens_scores_max = torch.tensor([compute_identifications_scores(scores_true_refs[nitr], scores_false_refs_max[nitr,:], summaryType='max', device=device) for nitr in range(nUniBests)]).to(device)
                scores_false_refs_mean = torch.vstack([step_summarize_scores(scores_false_refs[nitr,:],nCapEach, summaryType='mean', device=device) for nitr in range(nUniBests)]).to(device)
                idens_scores_mean = torch.tensor([compute_identifications_scores(scores_true_refs[nitr], scores_false_refs_mean[nitr,:], summaryType='mean', device=device) for nitr in range(nUniBests)]).to(device)
                print('%2s[false ref]::max:score = %.4f; mean:score = %.4f [t=%.5f]' % (btype,torch.mean(scores_false_refs_max[lastIdx,:]),torch.mean(scores_false_refs_mean[lastIdx,:]),time.time() - start)) if verbose else None
                print('%2s[iden acc] ::max:cr = %.4f%%; mean:score = %.4f%% [t=%.5f]' % (btype,idens_scores_max[lastIdx]*100,idens_scores_mean[lastIdx]*100,time.time() - start)) if verbose else None

                SCORES[btype] = {}
                SCORES[btype]['scores_true_refs'] = np.array(torch.vstack([scores_true_refs[i,:] for i in uni2org_idx]).to(device).cpu().detach())

                SCORES[btype]['scores_false_refs_org'] = np.array(torch.vstack([scores_false_refs[i,:] for i in uni2org_idx]).to(device).cpu().detach())

                SCORES[btype]['scores_false_refs_max'] = np.array(torch.vstack([scores_false_refs_max[i,:] for i in uni2org_idx]).to(device).cpu().detach())
 
                SCORES[btype]['scores_false_refs_mean'] = np.array(torch.vstack([scores_false_refs_mean[i,:] for i in uni2org_idx]).to(device).cpu().detach())
 
                SCORES[btype]['idens_scores_max'] = np.array(torch.vstack([idens_scores_max[i] for i in uni2org_idx]).to(device).cpu().detach())

                SCORES[btype]['idens_scores_mean'] = np.array(torch.vstack([idens_scores_mean[i] for i in uni2org_idx]).to(device).cpu().detach())
 
            
        # summarize scores with expantion
        if eval_metric in ['BLEU','METEOR','ROUGE-L','CIDEr']:
            SCORES[eval_metric]['scores_true_refs'] = np.array(torch.tensor([scores_true_refs[i] for i in uni2org_idx]).to(device).cpu().detach())

            SCORES[eval_metric]['scores_false_refs'] = np.array(torch.vstack([scores_false_refs[i,:] for i in uni2org_idx]).to(device).cpu().detach())

            SCORES[eval_metric]['idens_scores'] = np.array(torch.tensor([idens_scores[i] for i in uni2org_idx]).to(device).cpu().detach())

    return SCORES


def evaluate_nlp_metric_wrapper2(tokenizer, caps_preds, caps_alls, labels, eval_metrics, nCapEach, scorer_base, device):
    """
    Compute various NLP evaluation metrics on a given set of captions.
    This functoin take inputs for all captions at once and compared with preds assuming that the number of correponding refereces are the same for true and false candidates.

    Args:
        tokenizer (Tokenizer): Tokenizer for encoding and decoding captions.
        caps_preds (list): List of predicted captions for each video.
        caps_alls (list): List of referece captions for all videos.
        labels (list): List of labels for all videos.
        eval_metrics (list): List of evaluation metric names (e.g., 'BLEU', 'METEOR', 'ROUGE-L', 'CIDEr', 'BERTscore').
        nCapEach (int): Number of captions for each video.
        scorer_base (Model): BertScore model
        device (str): Device for computation.

    Returns:
        dict: A dictionary containing evaluation scores for each metric.
    """

    # initialize parameters
    do_nltkflag = 1
    SCORES = {}
    start = time.time()
    nCandAll = int(len(caps_alls)/nCapEach)
    nVidFalse = nCandAll-1
    nPredCaps = len(caps_preds)

    # prepare true/false reference indices
    ref_true_vid_indices_all = []
    ref_others_vid_indices_all = []
    ref_true_indices_all = []
    ref_others_indices_all = []
    for ix, videoidx in enumerate(labels):
        start_vid_idx = videoidx
        end_vid_idx = videoidx+1
        start_idx = nCapEach*videoidx
        end_idx = nCapEach*(videoidx+1)
        ref_true_vid_indices_all.append(range(start_vid_idx,end_vid_idx))
        ref_others_vid_indices_all.append([idx for idx in range(start_vid_idx) if idx < start_vid_idx] + [idx for idx in range(end_vid_idx, nCandAll)])
        ref_true_indices_all.append(range(start_idx,end_idx))
        ref_others_indices_all.append([idx for idx in range(start_idx) if idx < start_idx] + [idx for idx in range(end_idx, len(caps_alls))])

    # compute metrics
    for eval_metric in eval_metrics:
        SCORES[eval_metric] = {}

        # preparations
        if eval_metric in ['BLEU','METEOR'] and do_nltkflag:
            print('Processing nltk tokenization...')
            # prepare computing BLEU and METEOR using nltk
            # pass encode&decode to make case consistent.
            tokenized_preds = [word_tokenize(tokenizer.decode(tokenizer.encode(c, add_special_tokens=False))) for c in caps_preds]
            tokenized_alls = [word_tokenize(tokenizer.decode(tokenizer.encode(c, add_special_tokens=False))) for c in caps_alls]
            tokenized_alls = [tokenized_alls[i:i + nCapEach] for i in range(0, len(tokenized_alls), nCapEach)]
            do_nltkflag = 0

        elif eval_metric in ['ROUGE-L']:
            print('Preprocessing for ROUGE-L computation...')
            # construct inputs of rouge
            # Preprocess predicted captions
            preprocessed_preds = [tokenizer.decode(tokenizer.encode(c, add_special_tokens=False)) for c in caps_preds]
            index_preds = range(0,nPredCaps)
            pred_set = convert_index_and_captions_to_dict(index_preds, preprocessed_preds)
            # Preprocess all captions
            preprocessed_alls = [tokenizer.decode(tokenizer.encode(c, add_special_tokens=False)) for c in caps_alls]
            preprocessed_alls = [preprocessed_alls[i:i + nCapEach] for i in range(0, len(preprocessed_alls), nCapEach)]
            index_alls = flatten_list([[i]*nCapEach for i in range(0,nPredCaps)])

        elif eval_metric in ['CIDEr']:
            print('Preprocessing for CIDEr computation...')
            # construct inputs of cider
            # Preprocess reference and others captions
            preprocessed_alls_tmp = [tokenizer.decode(tokenizer.encode(c, add_special_tokens=False)) for c in caps_alls]
            preprocessed_alls_tmp2 = [preprocessed_alls_tmp[i:i + nCapEach] for i in range(0, len(preprocessed_alls_tmp), nCapEach)]
            preprocessed_alls = flatten_list([preprocessed_all*nPredCaps for preprocessed_all in preprocessed_alls_tmp2])
            index_all_refs = flatten_list([[i]*nCapEach for i in range(0,nCandAll*nPredCaps)])
            ref_all_set = convert_index_and_captions_to_dict(index_all_refs, preprocessed_alls)
            # Preprocess predicted captions
            preprocessed_preds = [tokenizer.decode(tokenizer.encode(c, add_special_tokens=False)) for c in caps_preds]*(nVidFalse+1)
            index_preds = range(0,nPredCaps*nCandAll)
            pred_set = convert_index_and_captions_to_dict(index_preds, preprocessed_preds)

        elif eval_metric in ['BERTscore']:
            # prepare inputs
            bs_p_inputs_all_ref = flatten_list([[c]*len(caps_alls) for c in caps_preds])
            bs_t_inputs_all_ref = caps_alls*nPredCaps
            bertscore_types = ['F1','P','R']

        # Processing
        print('%s:'%(eval_metric))
        if eval_metric in ['BLEU']:
            smoothfunc = bleu_score.SmoothingFunction().method2
            # compute BLEU scores
            scores_all_refs = torch.tensor([[bleu_score.sentence_bleu(tokenized_all, c, smoothing_function=smoothfunc) for tokenized_all in tokenized_alls] for c in tokenized_preds]).to(device)
            scores_true_refs = torch.tensor([scores_all_refs[ix,vid_idx] for ix, vid_idx in enumerate(ref_true_vid_indices_all)]) 
            scores_false_refs = torch.vstack([torch.tensor([scores_all_refs[ix,vid_idx] for vid_idx in vid_indices]) for ix, vid_indices in enumerate(ref_others_vid_indices_all)])
            print('[true ref] :score = %.4f [t=%.5f]'%(torch.mean(scores_true_refs), time.time() - start))
            print('[false ref]:score = %.4f [t=%.5f]' %(torch.mean(scores_false_refs),time.time() - start))
            idens_scores = torch.tensor([compute_identifications_scores(scores_true_refs[nitr], scores_false_refs[nitr,:], summaryType='max', device=device) for nitr in range(nPredCaps)]).to(device)
            print('[iden acc] :cr = %.4f%% [t=%.5f]' % (torch.mean(idens_scores) * 100, time.time() - start))

        elif eval_metric in ['METEOR']:
            # compute METEOR scores
            scores_all_refs = torch.tensor([[meteor_score.meteor_score(tokenized_all, c) for tokenized_all in tokenized_alls] for c in tokenized_preds]).to(device)
            scores_true_refs = torch.tensor([scores_all_refs[ix,vid_idx] for ix, vid_idx in enumerate(ref_true_vid_indices_all)]) 
            scores_false_refs = torch.vstack([torch.tensor([scores_all_refs[ix,vid_idx] for vid_idx in vid_indices]) for ix, vid_indices in enumerate(ref_others_vid_indices_all)])
            print('[true ref] :score = %.4f [t=%.5f]'%(torch.mean(scores_true_refs), time.time() - start))
            print('[false ref]:score = %.4f [t=%.5f]' %(torch.mean(scores_false_refs),time.time() - start))
            idens_scores = torch.tensor([compute_identifications_scores(scores_true_refs[nitr], scores_false_refs[nitr,:], summaryType='max', device=device) for nitr in range(nPredCaps)]).to(device)
            print('[iden acc] :cr = %.4f%% [t=%.5f]' % (torch.mean(idens_scores) * 100, time.time() - start))

        elif eval_metric in ['ROUGE-L']:
            # compute ROUGE-L scores
            scores_all_refs = torch.tensor(np.array([Rouge().compute_score(convert_index_and_captions_to_dict(index_alls, preprocessed_all*nPredCaps),pred_set)[1] for preprocessed_all in preprocessed_alls])).T.to(device)
            scores_true_refs = torch.tensor([scores_all_refs[ix,vid_idx] for ix, vid_idx in enumerate(ref_true_vid_indices_all)]) 
            scores_false_refs = torch.vstack([torch.tensor([scores_all_refs[ix,vid_idx] for vid_idx in vid_indices]) for ix, vid_indices in enumerate(ref_others_vid_indices_all)])
            print('[true ref] :score = %.4f [t=%.5f]'%(torch.mean(scores_true_refs), time.time() - start))
            print('[false ref]:score = %.4f [t=%.5f]' %(torch.mean(scores_false_refs),time.time() - start))
            idens_scores = torch.tensor([compute_identifications_scores(scores_true_refs[nitr], scores_false_refs[nitr,:], summaryType='max', device=device) for nitr in range(nPredCaps)]).to(device)
            print('[iden acc] :cr = %.4f%% [t=%.5f]' % (torch.mean(idens_scores) * 100, time.time() - start))

        elif eval_metric in ['CIDEr']:
            # compute CIDEr scores
            scores_all_refs = Cider().compute_score(ref_all_set,pred_set)[1].reshape(nCandAll,nPredCaps).T
            scores_true_refs = torch.tensor([scores_all_refs[ix,vid_idx] for ix, vid_idx in enumerate(ref_true_vid_indices_all)]) 
            scores_false_refs = torch.vstack([torch.tensor([scores_all_refs[ix,vid_idx] for vid_idx in vid_indices]) for ix, vid_indices in enumerate(ref_others_vid_indices_all)])
            print('[true ref] :score = %.4f [t=%.5f]'%(torch.mean(scores_true_refs), time.time() - start))
            print('[false ref]:score = %.4f [t=%.5f]' %(torch.mean(scores_false_refs),time.time() - start))
            idens_scores = torch.tensor([compute_identifications_scores(scores_true_refs[nitr], scores_false_refs[nitr,:], summaryType='max', device=device) for nitr in range(nPredCaps)]).to(device)
            print('[iden acc] :cr = %.4f%% [t=%.5f]' % (torch.mean(idens_scores) * 100, time.time() - start))

        elif eval_metric in ['BERTscore']:
            # compute bert scores for all videos
            P_all_refs, R_all_refs, F1_all_refs = scorer_base.score(bs_p_inputs_all_ref,bs_t_inputs_all_ref)
            P_all_refs, R_all_refs, F1_all_refs = P_all_refs.reshape((nPredCaps,len(caps_alls))).to(device), R_all_refs.reshape((nPredCaps,len(caps_alls))).to(device), F1_all_refs.reshape((nPredCaps,len(caps_alls))).to(device)

            # summarize true/false reference comparisons for BERT scores
            for btype in bertscore_types:
                if btype == 'F1':
                    scores_true_refs = torch.vstack([F1_all_refs[ix,idx] for ix, idx in enumerate(ref_true_indices_all)])
                    scores_false_refs = torch.vstack([torch.tensor([F1_all_refs[ix,idx] for idx in indices]) for ix, indices in enumerate(ref_others_indices_all)])
                elif btype == 'P':
                    scores_true_refs = torch.vstack([P_all_refs[ix,idx] for ix, idx in enumerate(ref_true_indices_all)]) 
                    scores_false_refs = torch.vstack([torch.tensor([P_all_refs[ix,idx] for idx in indices]) for ix, indices in enumerate(ref_others_indices_all)])
                elif btype == 'R':
                    scores_true_refs = torch.vstack([R_all_refs[ix,idx] for ix, idx in enumerate(ref_true_indices_all)]) 
                    scores_false_refs = torch.vstack([torch.tensor([R_all_refs[ix,idx] for idx in indices]) for ix, indices in enumerate(ref_others_indices_all)])

                scores_false_refs_max = torch.vstack([step_summarize_scores(scores_false_refs[nitr,:],nCapEach, summaryType='max', device=device) for nitr in range(nPredCaps)]).to(device)
                idens_scores_max = torch.tensor([compute_identifications_scores(scores_true_refs[nitr], scores_false_refs_max[nitr,:], summaryType='max', device=device) for nitr in range(nPredCaps)]).to(device)
                scores_false_refs_mean = torch.vstack([step_summarize_scores(scores_false_refs[nitr,:],nCapEach, summaryType='mean', device=device) for nitr in range(nPredCaps)]).to(device)
                idens_scores_mean = torch.tensor([compute_identifications_scores(scores_true_refs[nitr], scores_false_refs_mean[nitr,:], summaryType='mean', device=device) for nitr in range(nPredCaps)]).to(device)
                print('%2s[true  ref]::max:score = %.4f; mean:score = %.4f [t=%.5f]' % (btype,torch.mean(torch.max(scores_true_refs,dim=1).values),torch.mean(torch.mean(scores_true_refs,dim=1)),time.time() - start))
                print('%2s[false ref]::max:score = %.4f; mean:score = %.4f [t=%.5f]' % (btype,torch.mean(scores_false_refs_max),torch.mean(scores_false_refs_mean),time.time() - start))
                print('%2s[iden acc] ::max:cr = %.4f%%; mean:score = %.4f%% [t=%.5f]' % (btype,torch.mean(idens_scores_max)*100,torch.mean(idens_scores_mean)*100,time.time() - start))

                SCORES[btype] = {}
                SCORES[btype]['scores_true_refs'] = np.array(scores_true_refs.cpu().detach())
                SCORES[btype]['scores_false_refs_org'] = np.array(scores_false_refs.cpu().detach())
                SCORES[btype]['scores_false_refs_max'] = np.array(scores_false_refs_max.cpu().detach())
                SCORES[btype]['scores_false_refs_mean'] = np.array(scores_false_refs_mean.cpu().detach())
                SCORES[btype]['idens_scores_max'] = np.array(idens_scores_max.cpu().detach())
                SCORES[btype]['idens_scores_mean'] = np.array(idens_scores_mean.cpu().detach())


        # summarize scores with expantion
        if eval_metric in ['BLEU','METEOR','ROUGE-L','CIDEr']:
            SCORES[eval_metric]['scores_true_refs'] = np.array(scores_true_refs.cpu().detach())
            SCORES[eval_metric]['scores_false_refs'] = np.array(scores_false_refs.cpu().detach())
            SCORES[eval_metric]['idens_scores'] = np.array(idens_scores.cpu().detach())

    return SCORES


import random
import spacy
from itertools import permutations
def generate_unique_shuffled_texts(text, target_tag, num_texts=1, max_attempts=1000, remove_original=True, verbose=False):
    """
    Generate unique shuffled texts by shuffling words with a specific part-of-speech tag.

    Args:
        text (str): The input text.
        target_tag (str): The target part-of-speech tag for shuffling.
        num_texts (int, optional): The number of unique shuffled texts to generate. Default is 1.
        max_attempts (int, optional): The maximum attempts for shuffling. Default is 1000.
        remove_original (bool, optional): Whether to remove the original sentence from shuffled texts if matched. Default is True.
        verbose (bool, optional): Whether output warning or not.

    Returns:
        list: A list of unique shuffled texts.
    """
    def shuffle_words(target_words):
        shuffled_words = list(target_words)
        attempt = 0
        while shuffled_words == target_words and attempt < max_attempts:
            random.shuffle(shuffled_words)
            attempt += 1
        return shuffled_words
    
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    
    target_words = [token.text for token in doc if token.pos_ == target_tag]
    num_target_words = len(target_words)
    
    possible_permutations = list(permutations(target_words))
    #print(len(possible_permutations))
    #print(num_texts)
    #print(possible_permutations)
    
    
    if num_texts > len(possible_permutations) and verbose:
        #print(f"Warning: The number of requested texts is larger than the number of possible permutations [{len(possible_permutations)}].")
        #else:
        print(f"Warning: The number of possible permutations is {len(possible_permutations)}.")
    
    shuffled_texts = set()
    random.shuffle(possible_permutations)
    while len(shuffled_texts) < num_texts and possible_permutations:
        shuffled_target_words = shuffle_words(possible_permutations.pop())
        
        output_tokens = []
        target_index = 0
        
        for token in doc:
            if token.pos_ == target_tag:
                output_tokens.append(shuffled_target_words[target_index])
                target_index += 1
            else:
                output_tokens.append(token.text)
        
        shuffled_text = ' '.join(output_tokens)
        
        # TODO: correct according to inputs.
        punctuations_l = [',', ';', ':', '.', '!', '?', '-', ')', ']', '}']
        for punc in punctuations_l:
            shuffled_text = shuffled_text.replace(' '+punc, punc)
        punctuations_r = ['-', '(', '[', '{']
        for punc in punctuations_r:
            shuffled_text = shuffled_text.replace(punc+' ', punc)

        
        if remove_original and shuffled_text == text:
            continue  # Skip adding the original sentence
        
        shuffled_texts.add(shuffled_text)
    
    return list(shuffled_texts)


def compute_bertscore_matrix(hyp, ref, model, tokenizer, scorer_base, device):

    hyp_inp = tokenizer(hyp, padding=False, return_tensors="pt").to(device)
    ref_inp = tokenizer(ref, padding=False, return_tensors="pt").to(device)
    model.eval()

    with torch.no_grad():
        hyp_out = model(input_ids=hyp_inp['input_ids'],attention_mask=hyp_inp['attention_mask'],output_hidden_states=True).hidden_states
        ref_out = model(input_ids=ref_inp['input_ids'],attention_mask=ref_inp['attention_mask'],output_hidden_states=True).hidden_states    

    sim_HxR = []
    for l in range(model.config.num_hidden_layers):
        ref_embedding = ref_out[l][0][1:-1] # ignore cls/sep tokens
        ref_embedding.div_(torch.norm(ref_embedding, dim=-1).unsqueeze(-1))
        hyp_embedding = hyp_out[l][0][1:-1] # ignore cls/sep tokens
        hyp_embedding.div_(torch.norm(hyp_embedding, dim=-1).unsqueeze(-1))

        sim_HxR.append(torch.matmul(hyp_embedding, ref_embedding.transpose(0, 1)))

    hyp_tokenIDs = hyp_inp['input_ids'][0][1:-1] # ignore cls/sep tokens
    ref_tokenIDs = ref_inp['input_ids'][0][1:-1] # ignore cls/sep tokens
    
    return sim_HxR, hyp_tokenIDs, ref_tokenIDs

def compute_tokenwise_bertscore(hyp, ref, model, tokenizer, scorer_base, device, idf_dict=[]):
    #layerIdx = 17 # default of roberta-large: https://github.com/Tiiiger/bert_score/blob/master/bert_score/utils.py#L40
    #baseline_vals = scorer_base.baseline_vals[2] #[P,R,F] :https://github.com/Tiiiger/bert_score/blob/master/bert_score/rescale_baseline/en/roberta-large.tsv
    #sim = [(s-scorer_base.baseline_vals[2])/(1-scorer_base.baseline_vals[2]) for s in sim]

    sim_HxR, hyp_tokenIDs, ref_tokenIDs = compute_bertscore_matrix(hyp, ref, model, tokenizer, scorer_base=scorer_base, device=device)
    
    Rs = [torch.max(sim_HxR[l],dim=0).values for l in range(model.config.num_hidden_layers)]
    Ps = [torch.max(sim_HxR[l],dim=1).values for l in range(model.config.num_hidden_layers)]
    F1s = [2*torch.sum(Rs[l])*torch.sum(Ps[l])/(torch.sum(Rs[l])+torch.sum(Ps[l])) for l in range(model.config.num_hidden_layers)]

    if idf_dict:
        token_idfs = [torch.tensor([idf_dict[int(htokid)] for htokid in hyp_tokenIDs]).to(device) for l in range(model.config.num_hidden_layers)]
        #Ps_idf = [torch.mul(Ps[l],torch.tensor([idf_dict[int(htokid)] for htokid in hyp_tokenIDs]).to(device)) for l in range(model.config.num_hidden_layers)]
        #Rs_idf = [torch.mul(Rs[l],torch.tensor([idf_dict[int(rtokid)] for rtokid in ref_tokenIDs]).to(device)) for l in range(model.config.num_hidden_layers)]
        #F1s_idf = [2*torch.sum(Rs[l])*torch.sum(Ps[l])/(torch.sum(Rs[l])+torch.sum(Ps[l])) for l in range(model.config.num_hidden_layers)]
    else:
        token_idfs = []

    return Ps, Rs, F1s, token_idfs

def tensors_to_padded_numpy(tensor_list):
    """
    Convert a list of PyTorch tensors into a padded 3D NumPy array.

    Args:
        tensor_list (list of torch.Tensor): List of PyTorch tensors.

    Returns:
        np.ndarray: Padded 3D NumPy array.
    """
    # Find the maximum number of columns among the tensors
    max_cols = max(t.shape[1] for t in tensor_list)

    # Create a list of tensors with padding
    padded_tensors = [torch.cat([t.cpu(), torch.zeros(t.shape[0], max_cols - t.shape[1])], dim=1) for t in tensor_list]

    # Convert the list of padded tensors to a 3D NumPy array
    np_array_3d = np.stack([t.numpy() for t in padded_tensors], axis=0)

    return np_array_3d


from bert_score import utils as bs_util
import csv
import json
def compute_idf_from_caption_database(tokenizer,candDataSets,nthreads=4):
    # construct idf dictionary based on multiple caption databases

    # load database refs
    caps_data =  load_caption_data(candDataSets)

    idf_dict = bs_util.get_idf_dict(caps_data, tokenizer, nthreads=nthreads)
    return idf_dict


def load_caption_data(capDataSets):

    capDataSets = capDataSets if isinstance(capDataSets, list) else [capDataSets]

    caps_data = []
    for capdat in capDataSets:
        # assign file path
        if capdat in ['GCC_train']:
            filepath = capdata_path+'/Train-GCC-training.tsv'
        elif capdat in ['GCC_val']:
            filepath = capdata_path+'/Validation-GCC-1.1.0-Validation.tsv'
        elif capdat in ['MSCOCO_train']:
            filepath = capdata_path+'/captions_train2014.json'
        elif capdat in ['MSCOCO_val']:
            filepath = capdata_path+'/captions_val2014.json'
        elif capdat in ['MSRVTT']:
            filepath = capdata_path+'/videodatainfo_2017.json'
        elif capdat in ['ck20']:
            filepath = capdata_path+'/caption_ck20.csv'
        else:
            raise ValueError('Invalid dataset name')

        # get text data
        if capdat in ['GCC_train','GCC_val']:
            for dat in csv.reader(open(filepath),delimiter = '\t'):
                caps_data.append(dat[0]) # capcolidx = 0
        elif capdat in ['MSCOCO_train','MSCOCO_val']:
            for dat in json.load(open(filepath))['annotations']:
                caps_data.append(dat['caption'])
        elif capdat in ['MSRVTT']:
            for dat in json.load(open(filepath))['sentences']:
                caps_data.append(dat['caption'])
        elif capdat in ['ck20']:
            f = csv.reader(open(filepath, "r"), delimiter=",", doublequote=True, quotechar='"', skipinitialspace=True)
            header = next(f)
            caps_data = caps_data+[row[1] for row in f]

        else:
            raise ValueError('Invalid dataset name')

    return caps_data


import h5py

def prepare_norm_params(normparam_path, nlayers, suffix='', device='cpu'):
    # Function: prepare_norm_params
    # Description: Prepares normalization parameters.
    # Parameters:
    #   - normparam_path (str): Directory path for normalization parameters
    #   - nlayers (int): Number of layers.
    #   - device: Device for processing.

    # Returns:
    #   - feat_mu_all (list): List of mean values for normalization.
    #   - feat_sd_all (list): List of standard deviation values for normalization.

    # prepare normalize parameters
    feat_mu_all,feat_sd_all = [],[]
    for i in range(nlayers):
            dat_name = f"{normparam_path}/layer{str(i+1).zfill(2)}.mat"
            d =  h5py.File(dat_name,'r')
            feat_mu_all.append(torch.tensor(np.array(d['mu'+suffix]).T).to(device))
            feat_sd_all.append(torch.tensor(np.array(d['sd'+suffix]).T).to(device))
    return feat_mu_all, feat_sd_all

def text_optimization_steps(feat_target, feat_mu_all, feat_sd_all, model, tokenizer, skip_token_ids_mlm, model_lm, tokenizer_lm, skip_token_ids_lm, params, device):

    # initialization 2
    txts = [tokenizer.unk_token]
    last_score = -1
    best_cands,scores_all,scores_eval_all = [],[],[]
    mlmsflag = 1
    start = time.time()

    nitr = 0
    retrycnt = 0
    reflesh_th = params['reflesh_th'][1]
    while nitr < params['nItr']:
        if nitr == 0:
            skip_token_ids_lm_nounk = [tok_id for tok_id in skip_token_ids_lm if tok_id != tokenizer.unk_token_id]
            feat_all, inputs = compute_sentence_feature_patterns_wrapper(txts,model_lm, tokenizer_lm, skip_token_ids=skip_token_ids_lm_nounk, do_norm=params['do_norm'], feat_mu_all=feat_mu_all, feat_sd_all=feat_sd_all, device=device, layerIdx=params['layerIdx'], max_batch_samp=params['max_batch_samp'])
            scores_r, scores_r_reg, _ = compute_score(tokenizer_lm, inputs, feat_target, feat_all, params['mLayerType'], params['metricType'], skip_token_ids=skip_token_ids_lm_nounk, length_penalty_type=params['length_penalty_type'],length_penalty_w=params['length_penalty_w'])
            print('[%d]:%s:[score=%.4f, score_reg=%.4f][t=%.5f]'%(nitr,txts[0],scores_r,scores_r_reg,time.time() - start))
            best_cands.append(txts[0])
            scores_all.append(scores_r[0])
            scores_eval_all.append(scores_r_reg[0])
        else:
            txts = txtsx.copy()
            
        # prepare masked candidates
        inputs_masked_all = prepare_masked_candidateIDs(txts, tokenizer, device, nGram4Mask=params['nGram4Mask'], nMaskPerSentence=params['nMaskPerSentence'], add_insert_mask=params['add_insert_mask'], nMaskCands=params['nMaskCands'])

        # generate candidate sentences
        cand_sentences = generate_candidate_sentences_fromIDs_wrapper(inputs_masked_all, model, tokenizer, device, mlm_sampling_type=params['mlm_sampling_type'], topk=params['topk'], skip_token_ids=skip_token_ids_mlm, multiMaskType=params['multiMaskType'], max_batch_samp=params['max_batch_samp'],add_mask_removal=params['add_mask_removal'])
        #print(len(cand_sentences))

        # add original candidate and get unique sentence
        cand_sentences = list(set(cand_sentences + txts))
        #print(len(cand_sentences))

        # remove empty candidate, if any
        cand_sentences[:] = [sent for sent in cand_sentences if sent != tokenizer.decode(tokenizer.encode('',add_special_tokens=False))]
        print('Empty candidates removed.' if '' in cand_sentences else '', end='')
        #print(len(cand_sentences))

        # compute language model feature outputs for filled candidate sentences
        feat_all, inputs = compute_sentence_feature_patterns_wrapper(cand_sentences,model_lm, tokenizer_lm, skip_token_ids=skip_token_ids_lm, do_norm=params['do_norm'], feat_mu_all=feat_mu_all, feat_sd_all=feat_sd_all, device=device, layerIdx=params['layerIdx'], max_batch_samp=params['max_batch_samp'])

        # compute scores and sort
        scores_r, scores_r_reg, _ = compute_score(tokenizer_lm, inputs, feat_target, feat_all, params['mLayerType'], params['metricType'], skip_token_ids=skip_token_ids_lm, length_penalty_type=params['length_penalty_type'],length_penalty_w=params['length_penalty_w'])
        gc.collect(), torch.cuda.empty_cache()

        # compute MLM scoring scores
        if params['mlms_fix_weight'] != 0:
            mlm_scores_sum, mlm_scores_mean, mlm_scores_each, mlmsflag = compute_mlm_scores_wrapper(model,tokenizer, device, cand_sentences, nMax_MLMs_cands=params['nMax_MLMs_cands'], mlmscoreType=params['mlmscoreType'], mlmsflag=mlmsflag)
            mlm_scores_mean[torch.isnan(mlm_scores_mean)] = 0 # e.g., in the case of only a single token

            scores_eval = scores_r_reg+mlm_scores_mean*params['mlms_fix_weight']
            scores_eval_sorted, sortidx = torch.sort(scores_eval, descending = True)
            scores_r_sorted = torch.tensor([scores_r[si] for si in sortidx[:params['beamwidth']]])
            mlm_scores_mean_sorted = torch.tensor([mlm_scores_mean[si] for si in sortidx[:params['beamwidth']]])

            txtsx = [cand_sentences[si] for si in sortidx[:params['beamwidth']]]

        else:
            scores_eval = scores_r_reg
            scores_eval_sorted, sortidx = torch.sort(scores_eval, descending = True)
            scores_r_sorted = [scores_r[si] for si in sortidx[:params['beamwidth']]]
            txtsx = [cand_sentences[si] for si in sortidx[:params['beamwidth']]]

        if torch.any(torch.isnan(scores_r)):
            raise ValueError('NaN values found.')

        # keep results
        print('[%d]:%s:[score=%.4f, score_reg=%.4f][t=%.5f]'%(nitr+1,txtsx[0],scores_r_sorted[0],scores_eval_sorted[0],time.time() - start))
        best_cands.append(txtsx[0])
        scores_all.append(scores_r_sorted[0])
        scores_eval_all.append(scores_eval_sorted[0])

        # re-initialize if under threshold
        nitr += 1
        if params['do_reflesh'] and nitr % params['reflesh_th'][0] == 0 and scores_r_sorted[0] < reflesh_th:
            retrycnt += 1
            nitr = 0
            txts = [tokenizer.unk_token]
            if retrycnt % params['reflesh_th'][2] == 0:
                reflesh_th = reflesh_th-params['reflesh_th'][3]
                print('Lower the reflesh threshold to %.2f [%d]'%(reflesh_th,retrycnt))
            else:
                print('Retry. Current reflesh threshold is %.2f [%d]'%(reflesh_th,retrycnt))
            last_score = -1
            best_cands,scores_all,scores_eval_all = [],[],[]
            mlmsflag = 1

        if scores_r_sorted[0] >= 1-params['optimal_th']:
            print('Optimized: r > %.10f'%(1-params['optimal_th']))
            # fill the following
            best_cands.extend([txtsx[0]] * (params['nItr'] - nitr))
            scores_all.extend([scores_r_sorted[0]] * (params['nItr'] - nitr))
            scores_eval_all.extend([scores_eval_sorted[0]] * (params['nItr'] - nitr))
            break

    return best_cands, scores_all, scores_eval_all


