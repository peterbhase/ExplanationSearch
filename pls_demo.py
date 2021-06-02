
import argparse
import time
import torch
import numpy as np
import math
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from explanation_methods_and_eval.src.models.parallel_local_search import PLSExplainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type = int, default = 0, help = 'gpu id to use')
    parser.add_argument("--seed", default='0', type=int, help='')  
    # PLS args
    parser.add_argument('--explanation_sparsity', default=.2, type=float, help="proportion of tokens to be retained by the explanation")
    parser.add_argument('--num_search', default=1000, type=int)
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--num_restarts', default=10, type=int, help="number of searches to run in parallel, with different starting points")
    parser.add_argument('--objective', default="suff", type=str, help='sufficiency or comprehensiveness objective')
    
    # parse args and set cuda device
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu}")
    args.device = device
    torch.cuda.set_device(device)
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # in PLS, there's a np.exp(c / temp) where temp is 0, so ignore this warning. (temp can be changed to change explore/exploit dynamics)
    np.seterr(all="ignore") 

    # define weight of evidence function for use in objective_function    
    def np_woe(p):
        return np.log(p / (1-p))

    # define objective function here. will use this to define an f(x) objective function to give to PLSExplainer class
    def objective_function(args, explanation, always_keep_idx, task_model, orig_pred_prob, orig_input_kwargs):
        '''
        computes suff or comp objective on the current explanation                
        - explanation : np.ndarray of shape (r,d) where r is number of parallel runs and d is number of tokens (excluding always_keep tokens described next). 
        - always_keep_idx : binary array of len d', where d' is the number of tokens including tokens to be always kept, like special tokens
        - task_model: classifier that returns tuple with logits as SECOND element (after e.g. the loss)
        - orig_pred_prob: the predicted probability (for the predicted class) of the classifier on the current data point
        - orig_input_kwargs: dict with kwargs used to obtain orig_pred_prob, including 'attention_mask' item
        returns the suff/comp of task_model computed on document using attention_mask=mask, as well as the weight of evidence version of the objective
        ''' 
        # split input into batches if too many parallel runs (i.e. parallel states = parallel data points) to fit into memory
        num_batches = max(1, math.ceil(args.num_restarts / args.batch_size)) # num_restarts is num parallel runs
        num_tokens = orig_input_kwargs['input_ids'].size(-1)
        explanations = np.array_split(explanation, indices_or_sections=num_batches)
        obj_vals, obj_val_woes = [], []
        # repeat original data point batch_size times
        stacked_kwargs = {
            k : v.expand([args.batch_size] + list(v.shape)[-1:]).squeeze(-1).clone() for k,v in orig_input_kwargs.items() # clone to reallocate memory / squeeze() for labels special case
        }
        # get eligible for removal idx
        eligible_for_removal = torch.ones(num_tokens).to(args.device)
        eligible_for_removal[always_keep_idx] = 0.
        eligible_for_removal_idx = torch.where(eligible_for_removal)[0]
        for explanation in explanations:
            new_attention_mask = torch.tensor(explanation).long().to(args.device)
            stacked_kwargs['attention_mask'][:,eligible_for_removal_idx] = new_attention_mask
            outputs = task_model(**stacked_kwargs)
            pred_probs = torch.softmax(outputs[1], dim=-1)
            pred_prob = torch.gather(pred_probs, 1, stacked_kwargs['labels'].reshape(-1,1)).detach().cpu().numpy()
            if args.objective == 'suff':
                obj_val = orig_pred_prob - pred_prob
                obj_val_woe = np_woe(orig_pred_prob) - np_woe(pred_prob)
            if args.objective == 'comp':
                obj_val = -(orig_pred_prob - pred_prob)
                obj_val_woe = -(np_woe(orig_pred_prob) - np_woe(pred_prob))
            obj_vals.append(obj_val)
            obj_val_woes.append(obj_val_woe)
        obj_val = np.concatenate(obj_vals).reshape(-1)
        obj_val_woe = np.concatenate(obj_val_woes).reshape(-1)
        return obj_val, obj_val_woe

    # define some test data points. these are SNLI dev points
    data_points = [
        ("A dog swims in a pool .",    "A puppy is swiming ."),
        ("A man on a beach chopping coconuts with machete .",   "A man is outdoors"),                
        ("Young people sit on the rocks beside their recreational vehicle .",  "A group of senior citizens sit by their RV ."),
        ("After playing with her other toys , the baby decides that the guitar seems fun to play with as well .",   "A blonde baby ."),
        ('A teenage girl in winter clothes slides down a decline in a red sled .'  ,'A girl is by herself on a sled .'),
    ]
    # label_strs = ["neutral", "entailment", "contradiction", "neutral"]
    label_strs = ["neutral", "entailment", "contradiction", "neutral", "entailment"]
    id2label = {
        0: "entailment",
        1: "neutral",
        2: "contradiction"
    }
    label2id = {v:k for k,v in id2label.items()}
    label_ids = [label2id[label] for label in label_strs]

    # warning about what the Replace function is and what the blanks represent
    print("-------------------------------------")
    print("NOTE: the current Replace(x,e) function is the Attention Mask function")
    print("This function uses the binary explanation e as the attention mask over tokens")
    print("Blanks (like this: __) in the output of this script represent 0s in the input attention mask")
    print("Actual words in the script output are seen by the model")
    print("-------------------------------------")

    # load nli model
    print("Loading model...")
    hg_model_hub_name = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
    tokenizer = AutoTokenizer.from_pretrained(hg_model_hub_name)
    model = AutoModelForSequenceClassification.from_pretrained(hg_model_hub_name)
    model = model.to(args.device).eval()
    special_tokens = [tokenizer.bos_token_id, tokenizer.sep_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id]    

    # explain each data point
    for point_no, (data_point, label) in enumerate(zip(data_points, label_ids)):
        premise, hypothesis = data_point
        tokenized_input_seq_pair = tokenizer.encode_plus(premise, hypothesis, max_length=256, return_token_type_ids=True, truncation=True)
        model_kwargs = {
            'input_ids' : torch.Tensor(tokenized_input_seq_pair['input_ids']).long().unsqueeze(0).to(args.device),
            'token_type_ids' : torch.Tensor(tokenized_input_seq_pair['token_type_ids']).long().unsqueeze(0).to(args.device),
            'attention_mask': torch.Tensor(tokenized_input_seq_pair['attention_mask']).long().unsqueeze(0).to(args.device),
            'labels' : torch.Tensor([label]).long().to(args.device),
        }
        always_keep_idx = np.argwhere([_id in special_tokens for _id in model_kwargs['input_ids'].squeeze()]).reshape(-1)
        always_keep_idx = torch.Tensor(always_keep_idx).long().to(args.device)
        num_spcl_tokens = len(always_keep_idx)
        search_dimensionality = model_kwargs['input_ids'].shape[-1] - num_spcl_tokens

        outputs = model(**model_kwargs)
        pred_probs = torch.softmax(outputs[1], dim=1)[0]
        pred_class = torch.argmax(pred_probs)
        pred_prob = pred_probs[pred_class].item()

        def current_objective_function(explanation):
            return objective_function(
                args=args, 
                explanation=explanation, 
                always_keep_idx=always_keep_idx,
                task_model=model, 
                orig_pred_prob=pred_prob, 
                orig_input_kwargs=model_kwargs
            )

        # initialize PLS 
        PLS = PLSExplainer(
            objective_function=current_objective_function, 
            target_sparsity=args.explanation_sparsity, 
            eval_budget=args.num_search,
            dimensionality=search_dimensionality,
            restarts=args.num_restarts, # num parallel runs
            temp_decay=0, # temp in simulated annealing (0 <= x <= 1). set to higher value for more exploration early in search
            search_space='exact_k',
            no_duplicates=True # avoid duplicate evaluations of objective function
        )

        # run the search for an explanation
        print(f"Searching for explanation for point {point_no}...", end='\r')
        start = time.time()
        explanations, obj_values, obj_woe_values = PLS.run()
        print(f"Searching for explanation for point {point_no}...took {(time.time()-start):.2f} seconds!")

        best_explanation = explanations.tolist()[-1]
        best_obj = obj_values[-1]
        # flip sign on comp objective, since we negated in objective_function
        if args.objective == 'comp':
            best_obj *= -1

        input_ids = model_kwargs['input_ids'].squeeze().cpu().tolist()
        model_input_str = tokenizer.decode([_id for _id in input_ids if _id != tokenizer.pad_token_id])
        eligible_ids = [_id for i, _id in enumerate(input_ids) if i not in always_keep_idx]
        blank_id = tokenizer.encode(' __ ', add_special_tokens=False)[0]
        explanation_tokens = [_id if best_explanation[i] == 1 else blank_id for i, _id in enumerate(eligible_ids)]
        explanation_str = tokenizer.decode(explanation_tokens)

        print("Model input: ", model_input_str)
        print("Explanation: ", explanation_str)
        print(f"Model predicts {id2label[pred_class.item()]} with pred prob: {pred_prob:.3f} | Pred prob for explanation: {pred_prob-best_obj:.3f} | {args.objective}: {100*best_obj:.2f} points")
        print(f"Input label: {id2label[label]}")
        print()

