import os
import argparse
# import utils

def task_model(args):
    for seed in seeds:
        for model in ['bert-base-uncased']:
            for data in ['sst2']:
            # for data in ['boolq_raw', 'fever', 'multirc', 'esnli_flat', 'sst2', 'evidence_inference']:  
                for masking in [0]:
                    masking_str='_masked' if masking else ''
                    _masking='--masking_augmentation' if masking else ''
                    if data == 'esnli_flat':
                        n = 50000
                    else:
                        n = -1
                    e_name = f"{model[:9]}{masking_str}_sd{seed}"
                    n_epochs = 10
                    data_addin = f'--num_datapoints {n}' if n > 0 else ''
                    os.system(f"PYTHONPATH=./:$PYTHONPATH python src/scripts/task_script.py --cuda_device {args.gpu} --batch_size 4 --num_epochs {n_epochs} "
                              f"--task_model_name {model} --dataset {data} --exp_name {e_name} --lr 1e-5 {_masking} "
                              f"{data_addin} {small_data_addin} --seed {seed} "
                    )

def exact_counterfactual_training(args):
    for seed in seeds:
        for model in ['bert-base-uncased']:
            for data in ['esnli_flat', 'sst2']:
                masking_str='_exact-CT'
                _masking='--masking_augmentation --PLS_masks ' 
                if data == 'esnli_flat':
                    n = 50000
                else:
                    n = -1
                e_name = f"{model[:9]}{masking_str}_sd{seed}"
                n_epochs = 5 # see allennlp.training.trainer for PLS_masks args
                data_addin = f'--num_datapoints {n}' if n > 0 else ''
                os.system(f"PYTHONPATH=./:$PYTHONPATH python src/scripts/task_script.py --cuda_device {args.gpu} --batch_size 4 --num_epochs {n_epochs} "
                          f"--task_model_name {model} --dataset {data} --exp_name {e_name} --lr 1e-5 {_masking} "
                          f"{data_addin} {small_data_addin} --seed {seed} "
                )

def gradient_search(args):
    for seed in seeds:
        for controlled in [1]:
            for model in ['bert-base-uncased']:
                for data in ['boolq_raw', 'fever', 'multirc', 'esnli_flat', 'sst2', 'evidence_inference']:
                    for metric in ['suff','comp']:
                        for masking in [0, 1]:
                            masking_str='_masked' if masking else ''
                            task_model = f"{model[:9]}{masking_str}_sd{seed}"
                            n = 500
                            n_str = n if n > 0 else 'Full'
                            if controlled:
                                steps = 20
                                epochs = 22
                                cc = '_cc'
                            else:
                                steps = 20
                                epochs = 50
                                cc = ''
                            masking_str='_masked' if masking else ''
                            e_name = f"gradient_search{masking_str}_n{n_str}{cc}_{metric}_sd{seed}"
                            e_path = os.path.join('outputs', 'gradient_search', data, 'test', e_name + '_final_result.json')
                            if not os.path.exists(e_path) or eval_only != '':
                                print(f"\nStarting {e_name} on {data}\n")
                                os.system(f"PYTHONPATH=./:$PYTHONPATH python src/scripts/gradient_script.py --cuda_device {args.gpu} "
                                          f"--initialization random --top_k_selection up_to_k --objective suff --num_samples 1 --lr 1e-1 --objective {metric} "
                                          f"--steps_per_epoch {steps} --num_epochs {epochs} --sparsity_weight .001 --sparsity .05 --datasplit test "
                                          f"--task_model_name {model} --dataset {data} --exp_name {e_name} --task_model_exp_name {task_model} "
                                          f"{small_data_addin} --seed {seed} "
                                          f"--num_datapoints {n} {eval_only}"
                                          # f"--initialization gradient_search --eval_only "
                                          # f"--num_datapoints 200 --datasplit val --task_model_exp_name {task_model} --patience 100 "
                                )

def LIME(args):
    for seed in seeds:
        # controlled=0
        # for samples in [250]: #[76, 250]
        for controlled in [1]:
            for model in ['bert-base-uncased']:
                for data in ['sst2', 'fever', 'multirc', 'esnli_flat', 'boolq_raw', 'evidence_inference']:
                    for masking in [0, 1]:
                      n = 500
                      n_str = n if n > 0 else 'Full'
                      masking_str='_masked' if masking else ''
                      if controlled:
                          samples = 996
                          cc = '_cc'
                      else:
                          cc = f'_steps{samples}'
                      task_model = f"{model[:9]}{masking_str}_sd{seed}"
                      model_addin = '' if model == 'bert-base-uncased' else f"_{model[:9]}"
                      e_name = f"LIME{masking_str}_n{n_str}{cc}{model_addin}_sd{seed}"
                      e_path = os.path.join('outputs', 'lime', data, 'test', e_name + '_final_result.json')
                      if not os.path.exists(e_path) or eval_only != "":
                          print(f"\nStarting {e_name} on {data}\n")         
                          os.system(f"PYTHONPATH=./:$PYTHONPATH python src/scripts/lime_script.py --cuda_device {args.gpu} "
                                    f"--top_k_selection up_to_k --num_samples {samples} --batch_size 100 --threshold .5 --datasplit test "
                                    f"--task_model_name {model} --dataset {data} --exp_name {e_name} --task_model_exp_name {task_model} "
                                    f"{small_data_addin} --seed {seed} "
                                    f"--num_datapoints {n} {eval_only}"
                          )

def Anchors(args):
    for seed in seeds:
        # controlled=0
        for controlled in [1]:
            for model in ['bert-base-uncased']:
                for data in ['multirc']:
                # for data in ['boolq_raw', 'fever', 'multirc', 'esnli_flat', 'sst2', 'evidence_inference']:
                    for masking in [0, 1]:
                      n = 500
                      n_str = n if n > 0 else 'Full'
                      masking_str='_masked' if masking else ''
                      if controlled:
                          samples = 996
                          cc = '_cc'
                      else:
                          cc = f'_steps{samples}'
                      task_model = f"{model[:9]}{masking_str}_sd{seed}"
                      model_addin = '' if model == 'bert-base-uncased' else f"_{model[:9]}"
                      e_name = f"anchors{masking_str}_n{n_str}{cc}{model_addin}_sd{seed}"
                      e_path = os.path.join('outputs', 'anchors', data, 'test', e_name + '_final_result.json')
                      if not os.path.exists(e_path) or eval_only != "":
                          print(f"\nStarting {e_name} on {data}\n")
                          os.system(f"PYTHONPATH=./:$PYTHONPATH python src/scripts/anchors_script.py --cuda_device {args.gpu} "
                                    f"--num_samples {samples} --datasplit test "
                                    f"--task_model_name {model} --dataset {data} --exp_name {e_name} --task_model_exp_name {task_model} "
                                    f"{small_data_addin} --seed {seed} "
                                    f"--num_datapoints {n} {eval_only}"
                          )

def ordered_search(args):
    for seed in seeds:
        for model in ['bert-base-uncased']:
          for data in ['boolq_raw', 'fever', 'multirc', 'esnli_flat', 'sst2', 'evidence_inference']:
              for metric in ['suff','comp']:
                for base in ['LIME']:
                    for masking in [0, 1]:
                        n = 500
                        n_str = n if n > 0 else 'Full'
                        masking_str='_masked' if masking else ''
                        task_model = f"{model[:9]}{masking_str}_sd{seed}"
                        for base_search in [76, 250]:
                            here_search = 750 if base_search == 250 else 231
                            if base == 'LIME':
                              saliency_exp_name = f"LIME{masking_str}_n{n_str}_steps{base_search}_sd{seed}"
                              init = '--initialization lime'
                            elif base == 'IG':
                              saliency_exp_name = f"IG{masking_str}_n{n_str}_steps{base_search}_sd{seed}"
                              init = '--initialization integrated_gradient '
                            elif base == 'gradient_search-comp':
                              saliency_exp_name = f"gradient_search{masking_str}_n{n_str}_search{base_search}_comp_sd{seed}"
                              init = '--initialization gradient_search '
                            elif base == 'gradient_search-suff':
                              saliency_exp_name = f"gradient_search{masking_str}_n{n_str}_search{base_search}_suff_sd{seed}"                                  
                              init = '--initialization gradient_search '
                            e_name = f'budget{here_search}_' + saliency_exp_name
                            e_path = os.path.join('outputs', 'ordered', data, 'test', e_name + '_final_result.json')
                            if not os.path.exists(e_path):
                                print(f"\nStarting {e_name} on {data}\n")  
                                additive = '--additive' if ('LIME' in e_name or 'IG' in e_name) else ''                        
                                os.system(f"PYTHONPATH=./:$PYTHONPATH python src/scripts/ordered_script.py --cuda_device {args.gpu} "
                                          f" --num_to_search {here_search} --batch_size 100 --datasplit test {init} "
                                          f"--task_model_name {model} --dataset {data} --exp_name {e_name} --task_model_exp_name {task_model} --saliency_exp_name {saliency_exp_name} "
                                          f"{small_data_addin} --seed {seed} {additive} "
                                          f"--num_datapoints {n} --save_all_metrics "
                                )

def vanilla_gradient(args):
    for seed in seeds:
        for model in ['bert-base-uncased']:
            for data in ['boolq_raw', 'fever', 'multirc', 'esnli_flat', 'sst2', 'evidence_inference']:
                for masking in [0, 1]:
                    n = 500
                    n_str = n if n > 0 else 'Full'
                    masking_str='_masked' if masking else ''
                    task_model = f"{model[:9]}{masking_str}_sd{seed}"
                    e_name = f"vanilla_gradient{masking_str}_n{n_str}_sd{seed}"
                    e_path = os.path.join('outputs', 'vanilla_gradient', data, 'test', e_name + '_final_result.json')
                    if not os.path.exists(e_path):
                        print(f"\nStarting {e_name} on {data}\n")                              
                        os.system(f"PYTHONPATH=./:$PYTHONPATH python src/scripts/vanilla_gradient_script.py --cuda_device {args.gpu} "
                                  f"--datasplit test --mode input_embeddings --up_to_k "
                                  f"--task_model_name {model} --dataset {data} --exp_name {e_name} --task_model_exp_name {task_model} "
                                  f"{small_data_addin} --seed {seed} "
                                  f"--num_datapoints {n} "
                        )

def random_search(args):
    for seed in seeds:
        for model in ['bert-base-uncased']:
            for data in ['boolq_raw', 'fever', 'multirc', 'esnli_flat', 'sst2', 'evidence_inference']:
                for masking in [0, 1]:
                    for num_search in [1000]:
                        n = 500
                        n_str = n if n > 0 else 'Full'
                        masking_str='_masked' if masking else ''
                        task_model = f"{model[:9]}{masking_str}_sd{seed}"
                        model_addin = '' if model == 'bert-base-uncased' else f"_{model[:9]}"
                        e_name = f"random{masking_str}_n{n_str}_search{num_search}{model_addin}_sd{seed}"
                        e_path = os.path.join('outputs', 'random', data, 'test', e_name + '_final_result.json')
                        save_trajectories = "--save_all_metrics" if eval_only == "" else ""
                        if not os.path.exists(e_path) or eval_only != '':
                            print(f"\nStarting {e_name} on {data}\n")                              
                            os.system(f"PYTHONPATH=./:$PYTHONPATH python src/scripts/random_search_script.py --cuda_device {args.gpu} "
                                      f" --num_to_search {num_search} --batch_size 10 --datasplit test "
                                      f"--task_model_name {model} --dataset {data} --exp_name {e_name} --task_model_exp_name {task_model} "
                                      f"{small_data_addin} --seed {seed} --search_space exact_k "
                                      f"--num_datapoints {n} {save_trajectories} {eval_only}"
                            )

def integrated_gradients(args):
    for seed in seeds:
        for model in ['bert-base-uncased']:
            for controlled in [1]:
                for data in ['boolq_raw', 'fever', 'multirc', 'esnli_flat', 'sst2', 'evidence_inference']:
                    for masking in [0, 1]:
                        n = 500
                        masking_str='_masked' if masking else ''
                        n_str = n if n > 0 else 'Full'
                        task_model = f"{model[:9]}{masking_str}_sd{seed}"
                        if controlled:
                            steps = 498
                            cc = '_cc'
                        else:
                            steps = 500
                            cc = ''
                        e_name = f"IG{masking_str}_n{n_str}{cc}_sd{seed}"
                        e_path = os.path.join('outputs', 'integrated_gradient', data, 'test', e_name + '_final_result.json')
                        if not os.path.exists(e_path):
                            print(f"\nStarting {e_name} on {data}\n")                              
                            os.system(f"PYTHONPATH=./:$PYTHONPATH python src/scripts/integrated_gradient_script.py --cuda_device {args.gpu} "
                                      f"--top_k_selection up_to_k --steps {steps} --baseline mask --input_type one_hot --datasplit test "
                                      f"--task_model_name {model} --dataset {data} --exp_name {e_name} --task_model_exp_name {task_model} "
                                      f"{small_data_addin} --seed {seed} "
                                      f"--num_datapoints {n} "
                            )

def hotflip(args):
    for seed in seeds:
        for model in ['bert-base-uncased']:
            for controlled in [1]:
                for data in ['boolq_raw', 'fever', 'multirc', 'esnli_flat', 'sst2', 'evidence_inference']:
                    for masking in [0, 1]:
                        if controlled:
                            steps = 21
                            width = 3
                            cc = '_cc'
                        else:
                            steps = 50
                            width = 5
                            cc = ''
                        n = 500
                        masking_str='_masked' if masking else ''  
                        n_str = n if n > 0 else 'Full'
                        task_model = f"{model[:9]}{masking_str}_sd{seed}"
                        e_name = f"hotflip{masking_str}_n{n_str}{cc}_sd{seed}"
                        e_path = os.path.join('outputs', 'hotflip', data, 'test', e_name + '_final_result.json')
                        if not os.path.exists(e_path):
                            print(f"\nStarting {e_name} on {data}\n")  
                            os.system(f"PYTHONPATH=./:$PYTHONPATH python src/scripts/hotflip_script.py --cuda_device {args.gpu} "
                                      f" --num_steps {steps} --datasplit test --beam_width {width} --initialization random --batch_size 100 "
                                      f"--task_model_name {model} --dataset {data} --exp_name {e_name} --task_model_exp_name {task_model} "
                                      f"{small_data_addin} --seed {seed} "
                                      f"--num_datapoints {n} "
                            )

def exhaustive_search(args):
    for seed in seeds:
        for model in ['bert-base-uncased']:
            for data in ['sst2']:
                for masking in [0, 1]:
                    n = 500
                    n_str = n if n > 0 else 'Full'
                    masking_str='_masked' if masking else ''
                    task_model = f"{model[:9]}{masking_str}_sd{seed}"
                    e_name = f"exhaustive{masking_str}_n{n_str}{cc}_sd{seed}"
                    os.system(f"PYTHONPATH=./:$PYTHONPATH python src/scripts/exhaustive_script.py --cuda_device {args.gpu} "
                              f" --batch_size 100 --datasplit test "
                              f"--task_model_name {model} --dataset {data} --exp_name {e_name} --task_model_exp_name {task_model} "
                              f"{small_data_addin} --seed {seed} "
                              f"--num_datapoints {n} "
                    )

def parallel_local_search(args):
    for seed in seeds:
        for model in ['bert-base-uncased']:
            for data in ['esnli_flat']:
            # for data in ['boolq_raw', 'fever', 'multirc', 'esnli_flat', 'sst2', 'evidence_inference']:
                for masking in [2]:
                # for masking in [0, 1, 2]:
                    for num_search in [1000]:
                        n = 500
                        num_restarts = 10
                        temp_decay = 0 # update rule is: update state only if proposal is better (no random updating)
                        n_str = n if n > 0 else 'Full'
                        if masking == 1:
                          masking_str='_masked'
                        if masking == 2:
                          masking_str='_exact-CT'
                        else:
                          masking_str = ''
                        use_last = '' if masking != 2 else '--use_last ' # use last model, not best val epoch, for exact-CT
                        task_model = f"{model[:9]}{masking_str}_sd{seed}"
                        model_addin = '' if model == 'bert-base-uncased' else f"_{model[:9]}"
                        e_name = f"PLS{masking_str}_n{n_str}_search{num_search}{model_addin}_sd{seed}"
                        e_path = os.path.join('outputs', 'PLS', data, 'test', e_name + '_final_result.json')
                        if not os.path.exists(e_path):
                            print(f"\nStarting {e_name} on {data}\n")
                            os.system(f"PYTHONPATH=./:$PYTHONPATH python src/scripts/parallel_local_script.py --cuda_device {args.gpu} "
                                      f" --num_to_search {num_search} --batch_size 10 --datasplit test "
                                      f"--task_model_name {model} --dataset {data} --exp_name {e_name} --task_model_exp_name {task_model} "
                                      f"{small_data_addin} --seed {seed} "
                                      f"--num_restarts {num_restarts} --temp_decay {temp_decay} --search_space exact_k {use_last} "
                                      f"--num_datapoints {n} --save_all_metrics "
                            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", '-e', type=str) 
    parser.add_argument("--gpu", default='0', type=str) 
    parser.add_argument("--server", default='13', type=str) 
    parser.add_argument("--seeds", default=1, type=int) 
    parser.add_argument("--start", default=0, type=int) 
    parser.add_argument("--train_batch_size", default=4, type=int, help='')
    parser.add_argument("--grad_accumulation_factor", default=1, type=int, help='')
    parser.add_argument("--small_data", '-s', action='store_true')
    parser.add_argument("--eval_only", '-eval', action='store_true')
    args = parser.parse_args()
 
    # globals
    server = args.server
    small_data_addin = f' --num_epochs 2 --num_datapoints 10 ' if args.small_data else ''
    eval_only = '--eval_only' if args.eval_only else ''
    if args.start > 0: args.seeds = max(args.start+1, args.seeds)
    seeds = list(range(args.start, args.seeds))
    tbs = args.train_batch_size
    gaf = args.grad_accumulation_factor

    # experiments
    if args.experiment == 'task_model': task_model(args)
    if args.experiment == 'exact_counterfactual_training': exact_counterfactual_training(args)
    if args.experiment == 'gradient_search': gradient_search(args)
    if args.experiment == 'LIME': LIME(args)
    if args.experiment == 'anchors': Anchors(args)
    if args.experiment == 'integrated_gradients': integrated_gradients(args)
    if args.experiment == 'ordered_search': ordered_search(args)
    if args.experiment == 'hotflip': hotflip(args)
    if args.experiment == 'random_search': random_search(args)
    if args.experiment == 'exhaustive_search': exhaustive_search(args)
    if args.experiment == 'masking_model': masking_model(args)
    if args.experiment == 'parallel_local_search': parallel_local_search(args)
    if args.experiment == 'vanilla_gradient': vanilla_gradient(args)

