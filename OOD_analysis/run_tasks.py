import os
import argparse
import utils
import pandas as pd

def baseline(args):
    for seed in seeds:
        for data in ['eSNLI']:
            if data=='eSNLI': data_dir = 'data/eSNLI' 
            if data=='sst2': data_dir =  'data/sst2'
            for model in ['bert-base-uncased', 'roberta-base']:
                for n in [10000]:
                    if 'bert-base' in model:
                        rs = f'{data}_bert_10k'
                    if 'bert-large' in model:
                        rs = f'{data}_bert-large_10k'
                    if 'roberta-base' in model:
                        rs = f'{data}_roberta_10k'
                    if 'roberta-large' in model:
                        rs = f'{data}_roberta-large_10k'
                    _epochs = 20                
                    data_addin = f'--use_num_train {n}' if n > 0 else ''
                    os.system(f"python main.py --gpu {args.gpu} --data_dir {data_dir} --train_batch_size {tbs} --grad_accumulation_factor {gaf} --num_train_epochs {_epochs} "
                              f" --model {model} --max_seq_len 120 {data_addin} "
                              f"{small_data_addin} --seed {seed}  --result_sheet {rs} "
                    )

def joint_baseline(args):
    for seed in seeds:
        for data in ['eSNLI']:
            if data=='eSNLI': data_dir = 'data/eSNLI' 
            if data=='sst2': data_dir =  'data/sst2'
            for model in ['bert-base-uncased']:
                for n in [10000]:
                    for masking_style in ['attention', 'zero-vector', 'mask-token', 'slice-out']:
                        data_addin = f'--use_num_train {n}' if n > 0 else ''
                        _epochs = 10
                        sparsity = -1
                        os.system(f"python main.py --gpu {args.gpu} --data_dir {data_dir} --train_batch_size {tbs} --grad_accumulation_factor {gaf} --num_train_epochs {_epochs} "
                                  f" --model {model} --max_seq_len 120 {data_addin} --masking_style {masking_style} --masking_proportion {sparsity} "
                                  f"--result_sheet eSNLI_10k_joint{sparsity} " \
                                  f"{small_data_addin} --seed {seed}  "
                        )

def masked(args):
    for seed in seeds:
        for data in ['eSNLI']:
            if data=='eSNLI': data_dir = 'data/eSNLI' 
            if data=='sst2': data_dir =  'data/sst2'
            for model in ['roberta-base', 'bert-base-uncased']:
                for n in [10000]:
                    for masking_style in ['marginalize-v2', 'mask-token', 'attention', 'zero-vector', 'slice-out']:
                        for sparsity in [.2, .5, .8]:
                            if (model == 'roberta-large'): 
                                e_name = f'{data}_roberta-la_e20_b10_n{n}_maskNone_tr0_sd{seed}'
                                rs = f'{data}_roberta-large_10k'
                            if (model == 'roberta-base'): 
                                e_name = f'{data}_roberta-ba_e20_b10_n{n}_maskNone_tr0_sd{seed}'
                                rs = f'{data}_roberta_10k'
                            if (model == 'bert-base-uncased'): 
                                e_name = f'{data}_bert-base-_e20_b10_n{n}_maskNone_tr0_sd{seed}'
                                rs = f'{data}_bert_10k'
                            n_samples = 10 if masking_style != 'marginalize-v2' else 1
                            data_addin = f'--use_num_train {n}' if n > 0 else ''
                            use_num_dev = -1 
                            # check if already written
                            rs_df = pd.read_csv(os.path.join('result_sheets', rs + '.csv'))
                            rs_df.seed = rs_df.seed.astype(float)
                            rs_df.sparsity = rs_df.sparsity.astype(str)
                            rs_df.masking_style = rs_df.masking_style.astype(str)
                            print(f"Checking {model} | style {masking_style} | sparsity {sparsity} | seed {seed}")
                            written = sum((rs_df['seed'] == seed).to_numpy() * \
                                          (rs_df['sparsity'] == str(sparsity)).to_numpy() * \
                                          (rs_df['masking_style'] == str(masking_style)).to_numpy()
                                      ) > 9000
                            if not written:
                                print(f"\n\tstarting {model} | style {masking_style} | sparsity {sparsity} | seed {seed}\n")
                                os.system(f"python main.py --gpu {args.gpu} --data_dir {data_dir} --test_batch_size 100 --use_num_dev {use_num_dev} "
                                          f" --model {model} --max_seq_len 120 {data_addin} --num_samples {n_samples} --eval_masked_only true "
                                          f" --masking_style {masking_style} --masking_proportion {sparsity} --num_marginalize 10 -e {e_name} --result_sheet {rs} --do_train false "
                                          f"{small_data_addin} --seed {seed}  "
                                )

def joint_masked(args):
    for seed in seeds:
        for data in ['eSNLI']:
            if data=='eSNLI': data_dir = 'data/eSNLI' 
            if data=='sst2': data_dir =  'data/sst2'
            for model in ['bert-base-uncased']:
                for n in [10000]:
                    for sparsity in [.2, .5, .8]:
                        for source_sparsity in ['attention', 'mask-token', 'zero-vector', 'slice-out']:
                            mask_style = source_sparsity
                            read_sparsity_str = "_prop-1.0"
                            if (model == 'roberta-large'):             
                                e_name = f'{data}_roberta-la_e10_b10_n{n}_mask{mask_style}{read_sparsity_str}_sd{seed}'
                                rs = f'{data}_roberta_large_10k_mask{source_sparsity}{read_sparsity_str}'
                            if (model == 'roberta-base'):           
                                e_name = f'{data}_roberta-ba_e10_b10_n{n}_mask{mask_style}{read_sparsity_str}_sd{seed}'
                                rs = f'{data}_roberta_10k_mask{source_sparsity}{read_sparsity_str}'
                            if (model == 'bert-base-uncased'):
                                e_name = f'{data}_bert-base-_e10_b10_n{n}_mask{mask_style}{read_sparsity_str}_sd{seed}'
                                rs = f'{data}_bert_10k_mask{source_sparsity}{read_sparsity_str}'
                            n_samples = 10
                            data_addin = f'--use_num_train {n}' if n > 0 else ''
                            # check if already written
                            rs_path = os.path.join('result_sheets', rs + '.csv')
                            written=False
                            if os.path.exists(rs_path):
                                rs_df = pd.read_csv(rs_path)
                                rs_df.seed = rs_df.seed.astype(str)
                                rs_df.sparsity = rs_df.sparsity.astype(str)
                                rs_df.masking_style = rs_df.masking_style.astype(str)
                                print(f"Checking {model} | style {mask_style} | sparsity {sparsity} | seed {seed}")
                                written = sum((rs_df['seed'] == str(seed)).to_numpy() * \
                                              (rs_df['sparsity'] == str(sparsity)).to_numpy() * \
                                              (rs_df['masking_style'] == mask_style).to_numpy()
                                          ) > 0
                            if not written:
                                print(f"\n\tstarting {model} | style {mask_style} | sparsity {sparsity} | seed {seed}\n")                    
                                os.system(f"python main.py --gpu {args.gpu} --data_dir {data_dir} "
                                          f" --model {model} --max_seq_len 120 {data_addin} --num_samples {n_samples} --eval_masked_only true "
                                          f" --masking_style {mask_style} --masking_proportion {sparsity} -e {e_name} --result_sheet {rs} --do_train false "
                                          f"{small_data_addin} --seed {seed}  "
                                )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", '-e', type=str) 
    parser.add_argument("--gpu", default='0', type=str) 
    parser.add_argument("--seeds", default=1, type=int) 
    parser.add_argument("--start", default=0, type=int) 
    parser.add_argument("--train_batch_size", default=10, type=int, help='')
    parser.add_argument("--grad_accumulation_factor", default=1, type=int, help='')
    parser.add_argument("--small_data", '-s', action='store_true')
    args = parser.parse_args()
 
    # globals
    small_data_addin = f'-s -ss 11 --num_train_epochs 2 ' if args.small_data else ''
    seeds = list(range(args.start, args.seeds))
    tbs = args.train_batch_size
    gaf = args.grad_accumulation_factor

    # experiments
    if args.experiment == 'baseline': baseline(args)
    if args.experiment == 'joint_baseline': joint_baseline(args)
    if args.experiment == 'masked': masked(args)    
    if args.experiment == 'joint_masked': joint_masked(args)

