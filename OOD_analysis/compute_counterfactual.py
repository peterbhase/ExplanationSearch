import torch
from torch import nn
import transformers
import numpy as np
from scipy import stats

def random_mask(size, prop, pad_to=None, ignore_positions=None):
    # do not count first token, which will always be cls token
    array = np.ones(size)
    eligible_idx = np.arange(size) if ignore_positions is None else np.setdiff1d(np.arange(size), ignore_positions)
    non_spcl_size = len(eligible_idx)
    where_zeros = np.random.choice(eligible_idx, size=int(non_spcl_size*prop), replace=False)
    array[where_zeros] = 0
    if pad_to is not None:
        padding = np.ones(pad_to-size)
        array = np.concatenate([array, padding], axis=-1)
    return array

def _pad_seq(seq, pad_token_id, length):
    while len(seq) < length:
        seq.append(pad_token_id)
    return seq

def _pad_tensor(x, pad_token_id, length):
    short_by = length-x.size(-1)
    assert short_by >= 0
    padding = (pad_token_id*torch.zeros(list(x.shape[:-1]) + [length])[...,:short_by]).long().to(x.device)
    x = torch.cat((x, padding), dim=-1)
    return x


def compute_counterfactual(args, model, tokenizer, input_ids, attention_mask, labels, 
                           num_samples=None, num_marginalize=None, mlm_model=None):
    '''
    return model output and probabilities for counterfactual data points
    also returns ONE of num_samples masked input ids (the last one in the for loop) for viewing in main.py
    if labels are provided, return loss
    if num_samples is provided, return average probabilities across num_samples of random masks
    if num_marginalize is provided, assume that args.masking_style=='marginalization' and sample num_marginalization imputed samples per num_sample
    '''
    single_counterfactual = args.masking_style != 'marginalize-v2'
    num_samples = 1 if num_samples is None else num_samples
    batch_size = input_ids.size(0)
    seq_lens = attention_mask.sum(-1)
    max_seq_len = attention_mask.size(-1)
    num_classes = args.num_classes
    total_loss = 0

    # get random masks. these are fixed by the seed in main.py
    # 1 is keep, 0 is mask. will be 1 where orig sequence is pad_ids (i.e. padded with 1s)
    where_special_idx = np.array([np.argwhere([_id in [tokenizer.cls_token_id, tokenizer.sep_token_id] for _id in seq]) for seq in input_ids])
    random_masks = [[
        random_mask(size=int(seq_lens[i].item()), prop=args.masking_proportion, pad_to=max_seq_len, ignore_positions=where_special_idx[i]) 
        for _ in range(num_samples)] for i in range(batch_size)]
    random_masks = torch.tensor(random_masks).float().to(input_ids.device) # batch_size x num_samples x seq_length

    if num_marginalize is None:
        preds = torch.zeros(batch_size, num_samples)
        label_probs = torch.zeros(batch_size, num_samples)
    else:
        joint_logprobs = torch.zeros(batch_size, num_marginalize, num_classes).to(args.device)
    for i in range(num_samples):
        input_mask = random_masks[:,i,:]
        if num_marginalize is None:
            masked_ids, masked_input_embeds, masked_attention_mask, _ = get_masked_inputs(args, model, tokenizer, input_ids, attention_mask, input_mask)
            outputs = model(inputs_embeds=masked_input_embeds, attention_mask=masked_attention_mask, labels=labels)
            total_loss += outputs[0]
            logits = outputs[1]
            probs = torch.softmax(logits, dim=-1)
            preds[:, i] = torch.argmax(probs, dim=-1)
            label_probs[:,i] = torch.gather(probs, -1, labels.reshape(-1,1)).reshape(-1)
        elif num_marginalize is not None:
            assert num_samples==1
            split_size = args.test_batch_size
            for j in range(num_marginalize):
                masked_ids, masked_input_embeds, masked_attention_mask, sample_logprobs = get_masked_inputs(args, model, tokenizer, input_ids, attention_mask, input_mask)
                outputs = model(inputs_embeds=masked_input_embeds, attention_mask=masked_attention_mask, labels=labels)
                total_loss += outputs[0]
                logits = outputs[1]
                y_logprobs = torch.log_softmax(logits, dim=-1)
                joint_logprobs[:,j,:] = y_logprobs + sample_logprobs.unsqueeze(-1) # want sample_logprobs to broadcast over num_classes dim
            marginal_logprobs = torch.logsumexp(joint_logprobs, dim=1) # this is the actual marginalization over the num_marginalize dim
            preds = torch.argmax(marginal_logprobs, dim=-1)
            label_probs = torch.gather(marginal_logprobs, -1, labels.reshape(-1,1)).reshape(-1)
    if num_marginalize is None:
        preds = preds.reshape(batch_size, -1)
        label_probs = label_probs.reshape(batch_size, -1)
        return_preds, _ = stats.mode(preds.cpu().numpy(), axis=1)
        return_preds = return_preds.reshape(-1)
        mean_label_probs = torch.mean(label_probs, dim=-1)
        total_loss /= num_samples
    else:
        return_preds = preds.cpu().numpy()
        mean_label_probs = label_probs
        total_loss /= (num_samples * num_marginalize)
    return masked_ids, total_loss, return_preds, mean_label_probs

def get_masked_inputs(args, model, tokenizer, input_ids, attention_mask, input_mask=None):
    # masked_ids always swap tokens with mask tokens
    # if args.masking_proportion < 0, randomly sample from hard-coded values
    if args.masking_proportion < 0:
        use_sparsity = np.random.choice([.5, .8, .9, .95], size=1).item()
    else:
        use_sparsity = args.masking_proportion
    if input_mask is None:
        seq_lens = attention_mask.sum(-1)
        where_special_idx = np.array([np.argwhere([_id in [tokenizer.cls_token_id, tokenizer.sep_token_id] for _id in seq]) for seq in input_ids])
        input_mask = [random_mask(size=int(seq_lens[i].item()), prop=use_sparsity, pad_to=args.max_seq_len, ignore_positions=where_special_idx[i]) for i in range(input_ids.size(0))]
        input_mask = torch.tensor(input_mask).float().to(input_ids.device) # batch_size x max_seq_len
    masked_ids = input_ids.masked_fill((1-input_mask).bool(), tokenizer.mask_token_id)
    if args.masking_style in ['mask-token', 'pad-token', 'zero-vector', 'fractional']:
        input_embeds = get_new_input_embeds(args, model, tokenizer, input_ids, attention_mask, input_mask)
    if args.masking_style in ['slice-out']:
        input_ids, attention_mask = slice_out(args, tokenizer, input_ids, attention_mask, input_mask)
        input_embeds = get_embeds(model, input_ids)
    if args.masking_style in ['attention', 'attention-subnormal']:
        attention_mask = get_new_attention_mask(args, attention_mask, input_mask)
        input_embeds = get_embeds(model, input_ids)
    if args.masking_style in ['marginalize-v2']:
        input_ids, sample_logprobs = impute_inputs(args, tokenizer, input_ids, attention_mask, input_mask)
        input_embeds = get_embeds(model, input_ids)
    else:
        sample_logprobs = None
    return masked_ids, input_embeds, attention_mask, sample_logprobs

def get_embeds(model, input_ids):
    if hasattr(model, 'roberta'):
        embeds = model.roberta.embeddings(input_ids)
    elif hasattr(model, 'bert'):
        embeds = model.bert.embeddings(input_ids)
    else:
        assert False, "get_embeds not supported for current model class"
    return embeds

def impute_inputs(args, tokenizer, input_ids, attention_mask, input_mask):
    '''
    given input_ids and an input_mask that indicates which tokens to convert to [MASK] tokens, imputes sequences in those locations with sampled tokens
    uses arbitrary imputation order
    return imputed ids and logprobs for imputations 
    '''
    mlm_model = args.mlm_model
    batch_size = input_ids.size(0)
    masked_ids = input_ids.masked_fill(input_mask==0, tokenizer.mask_token_id)    
    where_masked = [np.argwhere(masked_ids[i].cpu().numpy()==tokenizer.mask_token_id).reshape(-1) for i in range(batch_size)]
    num_masked = np.array([len(x) for x in where_masked])
    # random shuffle of imputation orders
    imputation_orders = [idx[np.random.choice(np.arange(num_masked[i]), size=num_masked[i], replace=False)] for i, idx in enumerate(where_masked)]
    # pad with 0, which means the sequence is finished being imputed
    imputation_orders = [_pad_seq(idx.tolist(), 0, length=max(num_masked)) for idx in imputation_orders]
    imputation_idx = torch.tensor(imputation_orders).to(args.device)
    imputation_logprobs = torch.zeros(batch_size).to(args.device)
    for impute_num in range(max(num_masked)):
        impute_idx = imputation_idx[:, impute_num]
        outputs = mlm_model(input_ids=masked_ids, attention_mask=attention_mask)
        token_logits = outputs[0]
        impute_token_logits = torch.stack([token_logits[i, impute_idx[i], :] for i in range(batch_size)])
        logprobs = torch.log_softmax(impute_token_logits, dim=-1)
        sample_tokens = torch.multinomial(input=torch.softmax(impute_token_logits, dim=-1), num_samples=1)
        sample_logprobs = torch.gather(logprobs, 1, sample_tokens).reshape(-1)
        for i, idx in enumerate(impute_idx):
            if idx == 0: # 0 means all MASK tokens have been imputed
                continue
            masked_ids[i,idx] = sample_tokens[i]
            imputation_logprobs[i] += sample_logprobs[i]
        # for i in range(2):
        #     seq = tokenizer.decode([x for x in masked_ids[i] if x != tokenizer.pad_token_id])
        #     print(f"seq {i} : {seq}")
    return masked_ids, imputation_logprobs

def slice_out(args, tokenizer, input_ids, attention_mask, input_mask):
    # slice out of the encoded token ids. there are inconsistencies in pre and post tokenization/encoding lengths when slicing out of the list of string tokens    
    sliced_ids = [ids[torch.where(input_mask[i]==1)[0].reshape(-1)] for i, ids in enumerate(input_ids)]
    sliced_ids = [_pad_tensor(seq, tokenizer.pad_token_id, attention_mask.size(-1)) for seq in sliced_ids]
    input_ids = torch.stack(sliced_ids).long().to(attention_mask.device)
    num_removed = [int(num.item()) for num in attention_mask.size(-1) - input_mask.sum(-1)]
    attention_mask = [_pad_tensor(attn_mask[num_removed[i]:], tokenizer.pad_token_id, attention_mask.size(-1)) for i, attn_mask in enumerate(attention_mask)]
    attention_mask = torch.stack(attention_mask)
    return input_ids, attention_mask

def get_new_attention_mask(args, attention_mask, input_mask):
    new_attention_mask = attention_mask*input_mask
    return new_attention_mask

def get_new_input_embeds(args, model, tokenizer, input_ids, attention_mask, input_mask):
    if args.masking_style == 'mask-token':
        input_ids = input_ids.masked_fill((1-input_mask).bool(), tokenizer.mask_token_id)
        input_embeds = get_embeds(model, input_ids)
    if args.masking_style == 'zero-vector':
        input_embeds = get_embeds(model, input_ids)
        for i in range(input_ids.size(0)):
            remove_idx = torch.where(input_mask[i]==0)[0].reshape(-1)
            input_embeds[i, remove_idx, :] = 0
    return input_embeds
