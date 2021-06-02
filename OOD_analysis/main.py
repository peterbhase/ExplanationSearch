import os 
import torch
import numpy as np
import pandas as pd
import argparse
import time
import utils
import copy
from torch.utils.data import TensorDataset, DataLoader
from utils import str2bool
from utils import data_name_to_num_labels, get_make_data_function
from report import Report
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, BertForMaskedLM, RobertaForMaskedLM
from transformers import get_linear_schedule_with_warmup
from scipy import stats
import compute_counterfactual

def load_data(args, tokenizer):
    print("Loading data...")
    start_time = time.time()
    data_dir = args.data_dir + '_' + args.experiment_name if args.data_name == 'synthetic' else args.data_dir
    train_name, dev_name, test_name = utils.get_file_names(args.data_name)
    train_path = os.path.join(data_dir, train_name)
    dev_path = os.path.join(data_dir, dev_name)
    test_path = os.path.join(data_dir, test_name)
    make_data_function = get_make_data_function(args.data_name)
    train_dataset, train_info = make_data_function(args, tokenizer, file_path = train_path)
    dev_dataset, dev_info =     make_data_function(args, tokenizer, file_path = dev_path)
    test_dataset, test_info =   make_data_function(args, tokenizer, file_path = test_path) 
    load_time = (time.time() - start_time) / 60
    print(f"Loading data took {load_time:.2f} minutes")
    print("Data info:")
    for split_name, info in zip(['train','dev','test'], [train_info, dev_info, test_info]):
        n, n_classes, label_dist = info['n'], info['n_classes'], [round(100*x,2) for x in info['label_dist'].values()]
        print(f'  {split_name}: {n} points | {n_classes} classes | label distribution : {label_dist}')
    args.num_classes = n_classes
    train_dataloader = DataLoader(TensorDataset(*train_dataset), shuffle=True, batch_size=args.train_batch_size, pin_memory = True)    
    dev_dataloader = DataLoader(TensorDataset(*dev_dataset), shuffle=False, batch_size=args.test_batch_size, pin_memory = True)    
    test_dataloader = DataLoader(TensorDataset(*test_dataset), shuffle=False, batch_size=args.test_batch_size, pin_memory = True)    
    if args.eval_on_train:
        dev_dataloader, test_dataloader = train_dataloader, test_dataloader
    return train_dataloader, dev_dataloader, test_dataloader

def load_model(args, device, tokenizer, finetuned_path = None):
    print(f"\nLoading task model: {finetuned_path if finetuned_path is not None else args.model}\n")
    config = AutoConfig.from_pretrained(args.model, num_labels=data_name_to_num_labels[args.data_name], cache_dir=args.cache_dir)
    config.__dict__.update(args.__dict__)
    if 'subnormal' == args.masking_style:
        print("Using roberta model with subnormalized attention probabilities under attention mask")
        model_class = RobertaForSequenceClassification
    else:
        model_class = AutoModelForSequenceClassification
    model = model_class.from_pretrained(args.model, config=config, cache_dir = args.cache_dir)
    if finetuned_path is not None:
        model_state_dict = torch.load(finetuned_path, map_location=lambda storage, loc: storage) # args for preventing memory leakage across gpus
        model.load_state_dict(model_state_dict) 
    model.to(device)
    model.train()
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    print("Num trainable model params: %.2fm" % (sum([np.prod(p.size()) for p in model_parameters])/1e6))
    return model

def load_optimizer(args, model, lr, num_train_optimization_steps, names_params_list = None):
    model.train()
    # for local Bert-based models, if model.use_adaptors, p.requires_grad restricts only to adaptor parameters bc .train() called in load_model
    param_optimizer = [(n, p) for n, p in list(model.named_parameters()) if p.requires_grad] if names_params_list is None else names_params_list
    no_decay = ['bias', 'LayerNorm', 'layer_norm']
    optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                                    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
    return optimizer

def train_or_eval_epoch(args, epoch, device, dataloader, stats_dict, multi_gpu, model, optimizer, scheduler, tokenizer, split_name):
    '''
    runs one epoch. returns stats_dict. updates model parameters if training
    '''
    # init stat vars
    loss_sum = 0
    acc_sum = 0
    n_data_points = 0
    n_batches = len(dataloader)
    start_time = time.time()
    preds_list=[]    
    labels_list=[]
    per_point_stats = {'label_probs' : [], 'weight_of_evidence' : []}
    is_train = (split_name == 'train' and optimizer is not None)
    # eval on cfs only when testing robustness of a model (NOT training), which is in this condition
    forward_only_on_counterfactuals = (split_name!='train' and args.masking_style is not None and args.eval_masked_only)
    forward_only_on_real_data = (args.do_train and split_name=='dev')
    training_classifier = (is_train and args.do_train)
    if is_train: 
        model.train()
    else:
        model.eval()

    for step, batch in enumerate(dataloader):
        running_time = (time.time()-start_time)
        est_run_time = (running_time/(step+1)*n_batches)
        rolling_acc = 100*acc_sum/n_data_points if step>0 else 0
        print(f"  {split_name.capitalize()} | Batch: {step+1}/{n_batches} | Acc: {rolling_acc:.2f} |"
              f" Loss: {loss_sum/(step+1):.4f} | Speed: {running_time/(n_data_points+1):.3f} secs / point (est. remaining epoch time: {(est_run_time-running_time)/60:.2f} minutes)", end = '\r')
        
        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:

            # unpack batch vars
            batch = [item.to(device) for item in batch]
            data_idx, input_ids, attention_mask, labels = batch[:6]
            batch_size = input_ids.size(0)

            # make model inputs according to masking style
            masked_ids = None
            if not forward_only_on_counterfactuals:
                if args.masking_style is None or forward_only_on_real_data:
                    model_kwargs = {'input_ids' : input_ids,
                                    'attention_mask' : attention_mask,
                                    'labels' : labels
                    }
                else:
                    masked_ids, masked_input_embeds, masked_attention_mask, _ = compute_counterfactual.get_masked_inputs(args, model, tokenizer, input_ids, attention_mask)
                    doubled_labels = torch.cat((labels, labels))                    
                    orig_embeds = compute_counterfactual.get_embeds(model, input_ids)
                    input_embeds = torch.cat([orig_embeds, masked_input_embeds])
                    attention_mask = torch.cat([attention_mask, masked_attention_mask])
                    model_kwargs = {'inputs_embeds' : input_embeds,
                                    'attention_mask' : attention_mask,
                                    'labels' : doubled_labels
                    }
                outputs = model(**model_kwargs)
                loss = outputs[0] / args.grad_accumulation_factor
                logits = outputs[1]
                probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
                preds = np.argmax(probs, axis=-1)
                label_probs = torch.gather(torch.tensor(probs), -1, labels.cpu().reshape(-1,1))
                if len(preds) > batch_size:
                    preds = preds[:batch_size]

            elif forward_only_on_counterfactuals:
                masked_ids, loss, preds, label_probs = compute_counterfactual.compute_counterfactual(args, model, tokenizer, input_ids, attention_mask, labels, 
                        num_samples=args.num_samples, num_marginalize=args.num_marginalize if args.masking_style=='marginalize-v2' else None)

            # compute acc
            labels = labels.detach().cpu().numpy()
            weight_of_evidence = torch.log(label_probs / (1 - label_probs))
            labels_list.extend(labels.tolist())
            preds_list.extend(preds.tolist())
            per_point_stats['label_probs'].extend(label_probs.reshape(-1).tolist())
            per_point_stats['weight_of_evidence'].extend(weight_of_evidence.reshape(-1).tolist())
            acc_sum += np.sum(preds==labels)
            if multi_gpu:
                loss = loss.mean()
            # backward pass
            if is_train:
                loss.backward()
                if (step+1) % args.grad_accumulation_factor == 0 or (step == n_batches-1):
                    if training_classifier:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                        optimizer.step()
                        scheduler.step()
                    optimizer.zero_grad() 
            # track stats
            loss_sum += loss.item()
            n_data_points += len(labels)
            del loss
        
        if (args.print and epoch == 0 and step == 0 and is_train) or (args.print and epoch==-1 and step == 0 and split_name in ['train','dev']):
            print(f"\nEXAMPLE INPUTS")
            num_to_print = min(args.num_print, batch_size)
            for i in range(num_to_print):
                print(f"data point: {i} (idx: {data_idx[i].item()})")
                ids = [_id for _id in input_ids[i] if _id != tokenizer.pad_token_id]
                print(f"Orig input: {tokenizer.decode(ids, skip_special_tokens=False)}")
                if masked_ids is not None:
                    ids = [_id for _id in masked_ids[i] if _id != tokenizer.pad_token_id]               
                    print(f"Masked input: {tokenizer.decode(ids, skip_special_tokens=False)}")
                    print(f"Label vs pred: {labels[i]}, {preds[i]}")
                print()

    # summary stats
    loss_mean = loss_sum / n_data_points
    acc_mean = acc_sum / n_data_points
    if split_name == 'train': 
        stats_dict[f'{split_name}_loss'] = loss_mean
    stats_dict[f'{split_name}_acc'] = acc_mean * 100
    run_time = (time.time() - start_time) / 60
    print(f"  {split_name.capitalize()} time: {run_time:1.2f} min. ")

    # save eval statistics
    if epoch==-1 and split_name=='dev':
        if not os.path.exists('./result_sheets'):
            os.mkdir('result_sheets')
        sheet_path = os.path.join('result_sheets', args.result_sheet + '.csv')

        new_data = pd.DataFrame({
            'idx' : np.arange(len(dataloader.dataset)),
            'label' : labels_list,
            'pred' : preds_list,
            'seed' : args.seed,
            'masking_style' : args.masking_style if args.masking_style is not None else 'None',
            'sparsity' : args.masking_proportion if args.masking_style is not None else 'None',
            'label_probs' : per_point_stats['label_probs'],
            'weight_of_evidence' : per_point_stats['weight_of_evidence']
        })
        if args.fresh_result_sheet or not os.path.exists(sheet_path):
            data = new_data
        else:
            data = pd.read_csv(sheet_path)
            data = data.append(new_data, ignore_index=True)
        data.to_csv(sheet_path, index=False)
    
    return stats_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # misc. & debugging
    parser.add_argument('--gpu', type = int, default = 0, help = 'gpu id to use')
    parser.add_argument("--seed", default='0', type=int, help='')  
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--eval_on_train", action='store_true')
    parser.add_argument("--small_data", '-s', action='store_true')
    parser.add_argument("--small_size", '-ss', default=10, type=int, help='')
    parser.add_argument("--print", default = True, type=str2bool, help = 'flag for printing things helpful for debugging / seeing whats happening')
    parser.add_argument("--num_print", default=2, type=int, help='')
    # hyperparams
    parser.add_argument("--patience", default=100, type=int, help='after this many epochs with no dev improvement, break from training')
    parser.add_argument("--train_batch_size", default=10, type=int, help='')
    parser.add_argument("--test_batch_size", default=50, type=int, help='')
    parser.add_argument("--grad_accumulation_factor", default=1, type=int, help='')
    parser.add_argument("--num_train_epochs", default=20, type=int, help='')
    parser.add_argument("--lr", default=1e-5, type=float, help='')  
    parser.add_argument("--warmup_proportion", default=.1, type=float, help='')
    parser.add_argument("--max_x_len", default=100, type=int, help='max length of x in data loading.')  
    parser.add_argument("--max_e_len", default=40, type=int, help='max length of explanations as added to data.')
    parser.add_argument("--max_seq_len", default=140, type=int, help='max length of x plus whatever explanations may be appended to it')
    parser.add_argument("--max_grad_norm", default=1, type=float, help='')  
    # generic paths
    parser.add_argument("--report_dir", default='training_reports/', type=str)
    parser.add_argument("--save_dir", default='/playpen3/home/peter/saved_models/', type=str)
    parser.add_argument("--cache_dir", default='/playpen3/home/peter/cached_models/', type=str)
    parser.add_argument("--data_dir", default='data/eSNLI', type=str)    
    # experiment flags & paths
    parser.add_argument("--model", default='roberta-base', type=str, help='name of pretrained model')
    parser.add_argument("--pretrained_model", default = None, type=str, help = 'path for loading finetuned model')
    parser.add_argument("--pretrained_retriever", default = None, type=str, help = 'path for loading finetuned retriever')
    parser.add_argument("--use_percent_data", '-u', default=1., type=float, help='if < 1, use this percentage of the train data')
    parser.add_argument("--use_num_train", default=-1, type=int, help='if > 0, use this number of train data points')
    parser.add_argument("--use_num_dev", default=-1, type=int, help='if > 0, use this number of dev data points')
    parser.add_argument("--experiment_name", '-e', default = None, type=str, help = 'if not none, this overrides the experiment_name')
    parser.add_argument("--result_sheet", '-rs', default = None, type=str, help = 'if not none, this gives the result sheet name')
    parser.add_argument("--fresh_result_sheet", default = False, type=str2bool, help = 'refresh the result sheet (ie overwrite it')
    # masking arguments
    parser.add_argument("--masking_style", default = None, 
                        choices=[None, 'mask-token','fractional','attention','pad-token','zero-vector','slice-out', 'marginalize-v2', 'attention-subnormal'], type=str,
                        help="manner in which counterfactual model probabilities are computed. see compute_counterfactual.compute_counterfactual. \
                              Used during training if args.do_train is true!")
    parser.add_argument("--masking_proportion", default = -1, type=float,
                        help="sparsity of masks. proportion of tokens to be MASKED")
    parser.add_argument("--eval_masked_only", default = False, type=str2bool, help  = 'eval only on masked data distribution')
    parser.add_argument("--num_samples", default = 10, type=int, help  = 'num of random masks to apply. should be the same across seeds')
    parser.add_argument("--num_marginalize", default = 10, type=int, help='number of imputation samples to draw per random mask with masking_style==marginalize-v2')
    # control flow
    parser.add_argument("--do_train", default = True, type=str2bool, help = '')
    parser.add_argument("--do_eval", default = True, type=str2bool, help = '')
    parser.add_argument("--pre_eval", default = False, type=str2bool, help = '')

    # parse + experiment checks
    args = parser.parse_args()
    assert not (args.eval_masked_only and args.masking_style is None), "if eval on mask distr, please provide masking style"

    # add arguments + set model save dir
    args.data_name = os.path.split(args.data_dir)[-1]
    
    # GPU + SEED set-up
    n_gpu = torch.cuda.device_count()
    multi_gpu = (n_gpu > 1 and args.gpu == -1) # i.e. multiple gpus available and gpu choice not specified
    if multi_gpu: 
        device = torch.device("cuda") if args.gpu == -1 else torch.device(f'cuda:{args.gpu}')
        assert args.train_batch_size % n_gpu == 0, f"Train batch size will need to be allocated equally across {n_gpu} gpus, but {args.train_batch_size} cannot be"
        assert args.test_batch_size % n_gpu == 0, f"Eval batch size will need to be allocated equally across {n_gpu} gpus, but {args.test_batch_size} cannot be"
    else:
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(device)
    args.device = device
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    if multi_gpu:
        torch.cuda.manual_seed_all(args.seed)

    # make Report object, stats_dict, and paths
    args.experiment_name = experiment_name = utils.get_experiment_name(args) if args.experiment_name is None else args.experiment_name
    if args.small_data and args.do_train:
        experiment_name += f"_DEBUG"
    if args.small_data:
        args.result_sheet += "_DEBUG"
    model_path = os.path.join(args.save_dir, f'{experiment_name}.pth')
    print(f"\nStarting experiment: {experiment_name}") 
    # make pretrained_model_path
    pretrained_model_path = os.path.join(args.save_dir, f'{args.pretrained_model}.pth') if args.pretrained_model else None
    # report + stats
    report_name = f"report_{experiment_name}.txt"
    report_file = os.path.join(args.report_dir, report_name)
    if not os.path.exists(args.report_dir): os.mkdir(args.report_dir)
    score_names = ['train_loss', 'train_acc', 'dev_acc', 'test_acc']
    report = Report(args, report_file, experiment_name = experiment_name, score_names = score_names)
    stats_dict = {name : 0 for name in score_names}

    # LOAD TOKENIZER, DATA, MODEL, OPTIMIZER
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir = args.cache_dir)
    train_dataloader, dev_dataloader, test_dataloader = load_data(args, tokenizer)
    if args.do_train:
        model = load_model(args, device, tokenizer, finetuned_path = pretrained_model_path)
        num_train_optimization_steps = args.num_train_epochs * int(len(train_dataloader.dataset) / args.train_batch_size / args.grad_accumulation_factor)
        print("NUM OPT STEPS: ", num_train_optimization_steps)
        optimizer = load_optimizer(args, model = model, lr=args.lr, num_train_optimization_steps = num_train_optimization_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps= int(args.warmup_proportion * num_train_optimization_steps), num_training_steps=num_train_optimization_steps)
        if multi_gpu: 
            model = torch.nn.DataParallel(model, device_ids = range(n_gpu))
    if args.masking_style == 'marginalize-v2':
        if 'roberta' in args.model:
            args.mlm_model = RobertaForMaskedLM.from_pretrained('roberta-base', cache_dir=args.cache_dir).to(args.device)
        elif 'bert' in args.model:
            args.mlm_model = BertForMaskedLM.from_pretrained('bert-base-uncased', cache_dir=args.cache_dir).to(args.device)
    
    if args.debug:
        import pdb; pdb.set_trace()

    # pre-training checks
    if args.pre_eval: 
        print("Pre evaluation...")
        pre_stats_dict = train_or_eval_epoch(args=args,
                                            epoch=-1,
                                            device=device,
                                            stats_dict={},
                                            dataloader=dev_dataloader,
                                            multi_gpu=multi_gpu,
                                            model=model,
                                            optimizer=None,
                                            scheduler=None,
                                            tokenizer=tokenizer,
                                            split_name='dev',
        )
        report.print_epoch_scores(epoch = -1, scores = pre_stats_dict)

    # BEGIN TRAINING
    best_epoch = -1.0
    best_score = -1.0
    start_time = time.time()
    if args.do_train:
        print("\nBeginning training...\n")
        patience=0
        for e in range(args.num_train_epochs):
            print(f"Epoch {e} | LR: {optimizer.param_groups[0]['lr']:.6f}")
            stats_dict = train_or_eval_epoch(args=args,
                                        epoch=e,
                                        device=device,
                                        stats_dict=stats_dict,
                                        dataloader=train_dataloader,
                                        multi_gpu=multi_gpu,
                                        model=model,
                                        optimizer=optimizer,
                                        scheduler=scheduler,
                                        tokenizer=tokenizer,
                                        split_name='train',
            )
            stats_dict = train_or_eval_epoch(args=args,
                                            epoch=e,
                                            device=device,
                                            stats_dict=stats_dict,
                                            dataloader=dev_dataloader,
                                            multi_gpu=multi_gpu,
                                            model=model,
                                            optimizer=optimizer,
                                            scheduler=scheduler,
                                            tokenizer=tokenizer,
                                            split_name='dev',
            )
            # get score, write/print results, check for new best
            score = stats_dict['dev_acc']
            report.write_epoch_scores(epoch = e, scores = stats_dict)
            report.print_epoch_scores(epoch = e, scores = stats_dict)
            if score > best_score:
                print(f"  New best model. Saving model at {model_path}\n")
                torch.save(model.state_dict(), model_path)
                best_score, best_epoch = score, e
                patience = 0
            else:
                patience += 1
                if patience >= args.patience:
                    print(f"\n Patience of {args.patience} exceeded at epoch {e}! \n")
                    break
            
    end_time = time.time()
    training_time = (end_time-start_time) / 60
    unit = 'minutes' if training_time < 60 else 'hours'
    training_time = training_time if training_time < 60 else training_time / 60
    time_msg = f"\nTotal training time: {training_time:.2f} {unit}"
    print(time_msg)

    # FINAL EVAL
    model = load_model(args, device, tokenizer, finetuned_path = model_path)
    if multi_gpu: 
        model = torch.nn.DataParallel(model)
    
    # final evaluation
    if args.do_eval:
        print("\nGetting final eval results...\n")
        if args.data_name=='synthetic':
            stats_dict = train_or_eval_epoch(args=args,
                                            epoch=-1,
                                            device=device,
                                            stats_dict=stats_dict,
                                            dataloader=train_dataloader,
                                            multi_gpu=multi_gpu,
                                            model=model,
                                            optimizer=None,
                                            scheduler=None,
                                            tokenizer=tokenizer,
                                            split_name='train',
            )
        stats_dict = train_or_eval_epoch(args=args,
                                        epoch=-1,
                                        device=device,
                                        stats_dict=stats_dict,
                                        dataloader=dev_dataloader,
                                        multi_gpu=multi_gpu,
                                        model=model,
                                        optimizer=None,
                                        scheduler=None,
                                        tokenizer=tokenizer,
                                        split_name='dev',
        )
        if False:
            stats_dict = train_or_eval_epoch(args=args,
                                            epoch=-1,
                                            device=device,
                                            stats_dict=stats_dict,
                                            dataloader=test_dataloader,
                                            multi_gpu=multi_gpu,
                                            model=model,
                                            optimizer=None,
                                            scheduler=None,
                                            tokenizer=tokenizer,
                                            split_name='test',
            )
        train_acc = stats_dict['train_acc']
        dev_acc = stats_dict['dev_acc']
        test_acc = stats_dict['test_acc']
        final_msg = f"Best epoch: {best_epoch} | train acc: {train_acc:.2f} | " \
                    f"dev acc: {dev_acc:.2f} | test acc: {test_acc:.2f} "
        if args.do_train:
            report.write_final_score(args, final_score_str = final_msg, time_msg=time_msg)
        report.print_epoch_scores(epoch = best_epoch, scores = {k:v for k,v in stats_dict.items()})
