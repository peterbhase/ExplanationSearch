import argparse
import numpy as np
import pandas as pd
import torch
import copy
import os
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, RobertaTokenizer, T5Tokenizer
import time
import json

'''
general utilities
'''

def bootstrap(matrix: np.ndarray, n_samples=50000) -> dict:
    '''
    - matrix : ndarray of shape num_data_points x num_seeds, with elements representating binary correctness of individual predictions    
    returns a dict with
    - mean : mean correctness across data and seeds
    - CI : tuple with 2.5th and 97.5th percentiles of mean accuracy distribution under a block/grid bootstrap, where data and models are resampled with replacement
    - means : list of mean accuracies of each model
    '''
    return_dict = {
        'mean' : np.mean(matrix),
        'CI' : None,
        'means' : [np.mean(matrix[:,i]) for i in range(matrix.shape[-1])]
    }

    resampled_means = []
    n_rows = matrix.shape[0]
    n_cols = matrix.shape[-1]
    possible_row_idx = np.arange(n_rows)
    possible_col_idx = np.arange(n_cols)
    for i in range(n_samples):        
        rand_row_idx = np.random.choice(possible_row_idx, size=n_rows, replace=False)
        rand_col_idx = np.random.choice(possible_col_idx, size=n_cols, replace=False)
        sample_matrix = matrix[rand_row_idx, rand_col_idx]
        resampled_means.append(np.mean(sample_matrix))

    lb, ub = np.quantile(resampled_means, (.025, .975))
    return_dict['CI'] = (lb, ub)
    CI = (ub - lb) / 2
    print("Bootstrap results:")
    print(f"Ovr. mean: {return_dict['mean']*100:.2f} +/- {CI*100:.2f}")
    print(f"Individual means: {[round(x*100,2) for x in return_dict['means']]}")

    return return_dict


def weight_of_evidence(probs, labels):
    label_probs = torch.gather(probs, -1, labels.reshape(-1,1))
    c_label_probs = 1 - label_probs
    WoE = torch.log(label_probs / c_label_probs)
    return WoE

def make_experiment_sheet(experiment, params, num_seeds):
    sheet_folder = 'result_sheets'
    if not os.path.exists(sheet_folder): 
        os.mkdir(sheet_folder)
    sheet_path = os.path.join(sheet_folder, experiment + '.csv')
    cols = {param : [] for param in params}
    data = pd.DataFrame(cols)
    if not os.path.exists(sheet_path):
        data.to_csv(sheet_path, index=False)

def write_experiment_result(experiment, params):
    sheet_path = os.path.join('result_sheets', experiment + '.csv')
    data = pd.read_csv(sheet_path)
    dev_acc = round(np.load('tmp_dev_acc.npy').item(),3)
    test_acc = round(np.load('tmp_test_acc.npy').item(),3)
    params.update({f'dev_acc' : dev_acc})
    params.update({f'test_acc' : test_acc})
    new_data = data.append(params, ignore_index=True)
    new_data.to_csv(sheet_path, index=False)

def balanced_array(size, prop):
    # make array of 0s and 1s of len=size and mean ~= prop
    array = np.zeros(size)
    where_ones = np.random.choice(np.arange(size), size=int(size*prop), replace=False)
    array[where_ones] = 1
    if size==1:
        print("Asking for balanced array of size 1 could easily result in unintended behavior. This defaults to all 0s", end='\r')
    return array

def str2bool(v):
    # used for boolean argparse values
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def safe_decode(tokenizer, sequence, skip_special_tokens=True):
    return tokenizer.decode(filter(lambda x : x>=0, sequence), skip_special_tokens=skip_special_tokens)

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def _pad_seq(seq, length, pad_id):
    assert not len(seq) > length, "seq already too long"
    seq += [pad_id] * (length-len(seq))

def shuffle_lists(lists):
    [np.random.shuffle(_iter) for _iter in lists]

def dataloaders_equal(dataloader1, dataloader2):
    if len(dataloader1) != len(dataloader2): 
        return False
    for batch1, batch2 in zip(dataloader1, dataloader2):
        for item1, item2 in zip(batch1, batch2):
            if not torch.all(item1 == item2):
                print(item1)
                print(item2)
                return False
    return True


'''
get experiment name for writing files
'''
def get_experiment_name(args):
    if args.use_percent_data < 1:
        n_train = str(args.use_percent_data)
    elif args.use_num_train > 0:
        n_train = str(args.use_num_train)
    elif args.data_name == 'synthetic':
        n_train = str(args.num_train_synthetic)
    else:
        n_train = 'full'
    epochs = args.num_train_epochs
    mask_style = args.masking_style
    sparsity = args.masking_proportion
    sparsity_str = f'_prop{sparsity}' if args.masking_style is not None else '_tr0'
    batch_size = args.train_batch_size * args.grad_accumulation_factor
    e_name = f"{args.data_name}_{args.model[:10]}_e{epochs}_b{batch_size}_n{n_train}_mask{mask_style}{sparsity_str}" \
             f"_sd{args.seed}"
    return e_name

'''
dictionaries/functions for mapping related to models/datasets
'''

def get_file_names(data_name):
    if 'esnli' in data_name.lower():
        names = ['train.csv', 'dev.csv', 'test.csv']
    if data_name == 'sst2':
        names = ['train.jsonl', 'val.jsonl', 'test.jsonl']
    return names

def get_custom_model_class(name):
    if 't5' in name: 
        model_class = T5Wrapper
    elif 'roberta' in name:
        model_class = RobertaForSequenceClassification
    elif 'albert' in name:
        model_class = AlbertForSequenceClassification
    elif 'distilbert' in name:
        model_class = DistilBertForSequenceClassification
    elif 'bert' in name:
        model_class = BertForSequenceClassification
    return model_class

data_name_to_num_labels = {
    'eSNLI' : 3,
    'eSNLI_full' : 3,
    'sst2' : 2,
}

def get_make_data_function(name):
    data_name_to_make_data_function = {
        'eSNLI' : make_SNLI_data,
        'eSNLI_full' : make_SNLI_data,
        'sst2' : make_SST2_data
    }
    return data_name_to_make_data_function[name]


'''
text classification utilities 
'''

def make_SNLI_data(args, tokenizer, file_path):
    '''
    read SNLI, make data ready for DataLoader
    '''
    data = pd.read_csv(file_path)
    # locals
    split_name = os.path.split(file_path)[-1].split('.')[0]
    is_train = (split_name == 'train')
    label_map = {0: "neutral", 1: "entailment", 2: "contradiction"}
    # n, n_classes
    n_classes = 3
    n = len(data)
    # adjust size
    if args.small_data:
        n = int(args.small_size)
    elif args.use_percent_data < 1 and is_train: 
        n = int(n*args.use_percent_data)
    elif args.use_num_train > 0 and is_train:
        n = int(args.use_num_train)
    if args.use_num_dev > 0 and not is_train:
        n = int(args.use_num_dev)
    if is_train:
        random_idx = np.random.choice(np.arange(len(data)), size=n, replace=False)
        data = data.iloc[random_idx,:].reset_index()
    # load columns
    n_classes = len(label_map)
    ids = data['unique_key']
    premises = data['premise']
    hypotheses = data['hypothesis']
    labels = data['label']

    # make data
    data_idx = torch.arange(n) if is_train else -1*torch.ones(n)
    input_ids_list = []
    labels_list = []
    for i in range(n):
        premise = premises[i]
        hypothesis = hypotheses[i]
        premise_ids = tokenizer.encode(premise, add_special_tokens=False, max_length=args.max_seq_len, truncation=True)
        hypothesis_ids = tokenizer.encode(hypothesis, add_special_tokens=False, max_length=args.max_seq_len, truncation=True)
        label = labels[i]   
        # make classifier ids
        _truncate_seq_pair(premise_ids, hypothesis_ids, args.max_x_len-3)
        s_ids = [tokenizer.cls_token_id] + premise_ids + [tokenizer.sep_token_id] + hypothesis_ids + [tokenizer.sep_token_id]
        _pad_seq(s_ids, args.max_seq_len, tokenizer.pad_token_id)
        input_ids_list.append(s_ids)
        labels_list.append(label)
    input_ids = torch.tensor(input_ids_list).long()
    input_masks = (input_ids!=tokenizer.pad_token_id).float()
    labels = torch.tensor(labels_list).long()
    print(f"Example {split_name} inputs:")
    for i in range(args.num_print):
        print(f"({i})  y: {labels[i]} x: {tokenizer.decode(input_ids[i], skip_special_tokens=True)} ")
    return_data = [data_idx, input_ids, input_masks, labels]
    return_info = {'n' : n, 'n_classes' : n_classes, f'label_dist' : {i : sum(labels==i).item()/n for i in range(n_classes)}}
    return return_data, return_info


def make_SST2_data(args, tokenizer, file_path):
    '''
    read SNLI, make data ready for DataLoader
    '''
    data = pd.read_json(file_path, lines=True)
    # locals
    split_name = os.path.split(file_path)[-1].split('.')[0]
    is_train = (split_name == 'train')
    label_map = {0: "negative", 1: "positive"}
    # n, n_classes
    n_classes = 3
    n = len(data)
    # adjust size
    if args.small_data:
        n = int(args.small_size)
    elif args.use_percent_data < 1 and is_train: 
        n = int(n*args.use_percent_data)
    elif args.use_num_train > 0 and is_train:
        n = int(args.use_num_train)
    if is_train:
        random_idx = np.random.choice(np.arange(len(data)), size=n, replace=False)
        data = data.iloc[random_idx,:].reset_index()
    # load columns
    n_classes = len(label_map)
    sequences = data['passage']
    labels = data['answer']

    # make data
    data_idx = torch.arange(n) if is_train else -1*torch.ones(n)
    input_ids_list = []
    labels_list = []
    for i in range(n):
        seq = sequences[i]
        seq_ids = tokenizer.encode(seq, add_special_tokens=True, max_length=args.max_seq_len, truncation=True)
        label = labels[i]   
        _pad_seq(seq_ids, args.max_seq_len, tokenizer.pad_token_id)
        input_ids_list.append(seq_ids)
        labels_list.append(label)
    input_ids = torch.tensor(input_ids_list).long()
    input_masks = (input_ids!=tokenizer.pad_token_id).float()
    labels = torch.tensor(labels_list).long()
    print(f"Example {split_name} inputs:")
    for i in range(args.num_print):
        print(f"({i})  y: {labels[i]} x: {tokenizer.decode(input_ids[i], skip_special_tokens=True)} ")
    return_data = [data_idx, input_ids, input_masks, labels]
    return_info = {'n' : n, 'n_classes' : n_classes, f'label_dist' : {i : sum(labels==i).item()/n for i in range(n_classes)}}
    return return_data, return_info

