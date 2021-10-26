'''
this script gathers results from json files and computes confidence intervals and hypothesis tests
'''
import argparse
import os
import json
import numpy as np
import pandas as pd
import time

def safe_csv_load(path):
    if os.path.exists(path):    
        return pd.read_csv(path)

def p_value(betas):
    # calculate p-value for two-sided difference from 0 test with a bootstrapped distribution of statistics, beta  
    abs_mean_beta = np.abs(np.mean(betas))
    centered_betas = betas - np.mean(betas)
    outside_prop = np.mean(centered_betas < -abs_mean_beta) + np.mean(centered_betas > abs_mean_beta)  
    return outside_prop

def bootstrap_method(args, df, bootstrap_col_idx=None, bootstrap_row_idx=None, stats_dict = {}, stats_name=None):
    metrics = ['suff','suff_woe', 'comp', 'comp_woe']
    results = {}
    for metric in metrics:
        x = df[['idx', 'seed', metric]]
        x = x.pivot(index='idx', columns='seed', values=metric)
        x = x.to_numpy() # this will drop idx col
        n_rows, n_cols = x.shape
        if min(x.shape) == 0:
            continue
        means = []
        for i in range(args.num_samples):
            if bootstrap_col_idx is None:
                col_idx = np.random.choice(np.arange(n_cols), size=1 if not args.all_models else n_cols, replace=True)
            else:
                col_idx = bootstrap_col_idx[i]
            if bootstrap_row_idx is None:
                row_idx = np.random.choice(np.arange(n_rows), size=n_rows, replace=True)
            else:
                row_idx = bootstrap_row_idx[i]
            x_sample = x[row_idx, :]
            x_sample = x_sample[:, col_idx]
            mean = np.mean(x_sample)
            means.append(mean)
        lb, ub = np.quantile(means, (.025, .975))
        CI = (ub - lb) / 2
        ovr_mean = np.mean(x)
        if 'woe' in metric:
            mult_factor = 1
        else:
            mult_factor = 100
        result_str = f"{mult_factor*ovr_mean:.2f} ({mult_factor*CI:.2f})"
        results[metric] = result_str
        # add means to stats_dict
        stats_dict[f"{stats_name}_{metric}"] = np.array(means)
    return results

def bootstrap_results(args):
    '''
    uses several globals defined in __main__ block below
    '''
    stats_dict = {}
    dataframes = []
    for task_robust in ct_trained:
        masked_str = '_masked' if task_robust else ''
        cols = {'dataset' : [], 'num_seeds' : [], 'method' : [], 'task_robust' : [], 'suff' : [], 'comp' : [], 'suff_woe' : [], 'comp_woe' : []}
        ovr_data = pd.DataFrame(cols)
        ovr_data_save_name = f"bootstrapped_robust{task_robust}_explanation_results.csv"
        for dataset in datasets:
            for method in methods:                
                cols = {'idx' : [], 'seed' : [], 'suff' : [], 'comp' : [], 'suff_woe' : [], 'comp_woe' : []}
                method_data = pd.DataFrame(cols)
                for seed in seeds:
                    if method == 'lime':
                        e_name = f'LIME{masked_str}_n{n}{cc}{model_addin}_sd{seed}_per_point_metrics.csv'
                        path = os.path.join('outputs', method, dataset, split, e_name)
                        results = safe_csv_load(path)
                        if results is not None:
                            results['idx'] = np.arange(n)
                            results['seed'] = seed
                            method_data = method_data.append(results, ignore_index=True)
                    if method == 'gradient_search':
                        e_name = f'gradient_search{masked_str}_n{n}{cc}_suff_sd{seed}_per_point_metrics.csv'
                        path = os.path.join('outputs', method, dataset, split, e_name)
                        add_results = safe_csv_load(path)
                        if add_results is not None:
                            add_results = add_results[['id','suff','suff_woe']]                            
                            e_name = f'gradient_search{masked_str}_n{n}{cc}_comp_sd{seed}_per_point_metrics.csv'
                            path = os.path.join('outputs', method, dataset, split, e_name)                    
                            add_results2 = safe_csv_load(path)
                            if add_results2 is not None:
                                add_results['comp'] = add_results2['comp']
                                add_results['comp_woe'] = add_results2['comp_woe']
                                add_results['idx'] = np.arange(n)
                                add_results['seed'] = seed
                        method_data = method_data.append(add_results, ignore_index=True)
                    if method == 'integrated_gradient':
                        e_name = f"IG{masked_str}_n{n}{cc}{model_addin}_sd{seed}_per_point_metrics.csv" 
                        path = os.path.join('outputs', method, dataset, split, e_name)
                        results = safe_csv_load(path)
                        if results is None:
                            e_name = f"IG{masked_str}_n{n}_sd{seed}_per_point_metrics.csv" 
                            path = os.path.join('outputs', method, dataset, split, e_name)
                        results = safe_csv_load(path)
                        if results is not None:
                            results['idx'] = np.arange(n)
                            results['seed'] = seed
                            method_data = method_data.append(results, ignore_index=True)
                    if 'ordered_search-lime' in method:
                            method_dir = 'ordered'
                            if method == 'ordered_search-lime-250':
                                method_dir = 'ordered'
                                e_name = f'budget231_LIME{masked_str}_n{n}_steps76_sd{seed}_per_point_metrics.csv'
                            if method == 'ordered_search-lime-1000':
                                e_name = f'budget750_LIME{masked_str}_n{n}_steps250_sd{seed}_per_point_metrics.csv'
                            path = os.path.join('outputs', method_dir, dataset, split, e_name)
                            results = safe_csv_load(path)
                            if results is not None:
                                results['idx'] = np.arange(n)
                                results['seed'] = seed
                                method_data = method_data.append(results, ignore_index=True)
                    if method == 'hotflip':
                        method_dir = 'hotflip'
                        e_name = f'hotflip{masked_str}_n{n}{cc}{model_addin}_sd{seed}_per_point_metrics.csv'
                        path = os.path.join('outputs', method_dir, dataset, split, e_name)
                        results = safe_csv_load(path)
                        if results is not None:
                            results['idx'] = np.arange(n)
                            results['seed'] = seed
                            method_data = method_data.append(results, ignore_index=True)   
                    if method == 'random_search-250':
                        method_dir = 'random'
                        e_name = f'random{masked_str}_n{n}_search250{model_addin}_sd{seed}_per_point_metrics.csv'
                        path = os.path.join('outputs', method_dir, dataset, split, e_name)
                        results = safe_csv_load(path)
                        if results is not None:
                            results['idx'] = np.arange(n)
                            results['seed'] = seed
                            method_data = method_data.append(results, ignore_index=True)     
                    if method == 'random_search-1000':
                        method_dir = 'random'
                        e_name = f'random{masked_str}_n{n}_search1000{model_addin}_sd{seed}_per_point_metrics.csv'
                        path = os.path.join('outputs', method_dir, dataset, split, e_name)
                        results = safe_csv_load(path)
                        if results is not None:
                            results['idx'] = np.arange(n)
                            results['seed'] = seed
                            method_data = method_data.append(results, ignore_index=True)               
                    if method == 'exhaustive':
                        method_dir = 'exhaustive'
                        e_name = f'exhaustive{masked_str}_n{n}_sd{seed}_per_point_metrics.csv'
                        path = os.path.join('outputs', method_dir, dataset, split, e_name)
                        results = safe_csv_load(path)
                        if results is not None:
                            results['idx'] = np.arange(n)
                            results['seed'] = seed
                            method_data = method_data.append(results, ignore_index=True)
                    if method == 'anchors':
                        method_dir = 'anchors'
                        e_name = f'anchors{masked_str}_n{n}_cc_sd{seed}_per_point_metrics.csv'
                        path = os.path.join('outputs', method_dir, dataset, split, e_name)
                        results = safe_csv_load(path)
                        if results is not None:
                            results['idx'] = np.arange(n)
                            results['seed'] = seed
                            method_data = method_data.append(results, ignore_index=True)
                    if method == 'vanilla_gradient':
                        method_dir = f'{method}'
                        e_name = f'{method}{masked_str}_n{n}_sd{seed}_per_point_metrics.csv'
                        path = os.path.join('outputs', method_dir, dataset, split, e_name)
                        results = safe_csv_load(path)
                        if results is not None:
                            results['idx'] = np.arange(n)
                            results['seed'] = seed
                            method_data = method_data.append(results, ignore_index=True)
                    if method == 'PLS':
                        method_dir = f'PLS'
                        e_name = f'{method}{masked_str}_n{n}_search1000{model_addin}_sd{seed}_per_point_metrics.csv'
                        path = os.path.join('outputs', method_dir, dataset, split, e_name)
                        results = safe_csv_load(path)
                        if results is not None:
                            results['idx'] = np.arange(n)
                            results['seed'] = seed
                            method_data = method_data.append(results, ignore_index=True)

                # bootstrap for the method
                if 'suff_pred_prob' in method_data.columns:
                    method_data = method_data.drop(['suff_pred_prob', 'comp_pred_prob'], axis=1)
                method_data = method_data.dropna()
                if len(method_data) > 0:
                    stats_name = f'{dataset}_{method}{masked_str}'
                    results = bootstrap_method(args, method_data, bootstrap_col_idx=bootstrap_col_idx, bootstrap_row_idx=bootstrap_row_idx, stats_dict=stats_dict, stats_name=stats_name)
                    results['num_seeds'] = len(set(method_data['seed']))
                    results['method'] = method
                    results['task_robust'] = task_robust
                    results['dataset'] = dataset
                    ovr_data = ovr_data.append(results, ignore_index=True)

        masked = 'masked' if task_robust else 'non-masked'
        save_path = os.path.join('outputs', ovr_data_save_name)
        ovr_data.to_csv(save_path, index=False)
        dataframes.append(ovr_data)

    if len(dataframes) > 0:
        print("\nCombined data (masked columns interleaved)")
        combined_df = pd.concat(dataframes, keys=['dataset', 'method'], axis=1)
        if len(ct_trained) > 1:
            # combined_df = combined_df.iloc[:,[0,1,9,2,4,-4,5,-3,6,-2,7,-1]]
            combined_df = combined_df.iloc[:,[0,1,9,2,4,-4,5,-3]] # drop woe columns
        print(combined_df)
        combined_df.to_csv(os.path.join('outputs', 'all_explanation_results.csv'), index=False)

    # conduct hypothesis tests
    print('\n Hypothesis tests:')
    for test in hypothesis_tests:
        compare_name = test[0]
        baseline_name = test[1]
        if compare_name in stats_dict.keys():
            compare_means = stats_dict[compare_name]
            baseline_means = stats_dict[baseline_name]
            mean = np.mean(compare_means) - np.mean(baseline_means)
            diffs = compare_means - baseline_means
            p_val = p_value(diffs)
            lb, ub = np.quantile(diffs, (.025, .975))
            CI = (ub - lb) / 2
            compare = 'PLS' if 'PLS' in compare_name else '?'
            baseline = 'random' if 'random' in baseline_name else 'lime'
            mult_factor = 1 if 'woe' in baseline_name else 100
            print(f"Dataset: {compare_name.split('_')[0]:9s} | {compare:2s} vs {baseline:8s} | metric: {baseline_name.split('_')[-1]:8s} | masked : {str('masked' in compare_name):5s}" \
                    f" | Diff : {mult_factor*np.mean(diffs):+.2f} +/- {mult_factor*CI:.2f} (p = {p_val:.4f})" \
                    f" | Diff2 : {mult_factor*mean:+.2f} (mean1 : {mult_factor*np.mean(compare_means):.2f} mean2: {mult_factor*np.mean(baseline_means):.2f})"
                )
        else:
            print(f"Results not found for {compare_name} vs {baseline_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--all_models", default=True, action='store_true', help='default is to sample one arbitrary model, not resample all models') 
    parser.add_argument("--num_samples", '-n', type=int, default=100000, help='num bootstrap samples')
    args = parser.parse_args()

    # script parameters
    # methods = ['lime', 'gradient_search', 'vanilla_gradient', 'integrated_gradient', 'ordered_search-lime-250', 'ordered_search-lime-1000', \
    #             'hotflip', 'random_search-250', 'random_search-1000', 'PLS', 'exhaustive']
    methods = ['lime', 'random_search-1000', 'PLS', 'anchors']
    datasets = ['esnli_flat', 'boolq_raw', 'evidence_inference', 'fever', 'multirc', 'sst2']
    # datasets = ['esnli_flat']
    splits = ['test']
    metrics = ['suff','comp']
    # metrics = ['suff','suff_woe','comp','comp_woe']
    seeds = list(range(10))
    cc = '_cc' # compute controlled
    hypothesis_tests = []
    split = 'test'
    n = 500
    model = 'bert-base'
    # model = 'roberta-base'
    model_addin = '' if model=='bert-base' else f"_{model[:9]}"
    ct_trained = [0,1] if model == 'bert-base' else [1]

    # pre-generate bootstrap idx
    max_seed = max(seeds)
    # can set to None if not running hypothesis tests
    # bootstrap_col_idx = None
    # bootstrap_row_idx = None
    # use when running hypothesis tests, for same idx across runs
    bootstrap_col_idx = [np.random.choice(seeds, size=len(seeds), replace=True) for i in range(args.num_samples)]
    bootstrap_row_idx = [np.random.choice(np.arange(n),  size=n,  replace=True)  for i in range(args.num_samples)]
    
    # add pair-wise hypothesis tests to run
    if 'PLS' in methods:
        for dataset in datasets:
            for metric in metrics:
                for masked_str in ['', '_masked']:
                    PLS_name = f'{dataset}_PLS{masked_str}_{metric}'
                    for baseline in ['lime', 'random_search-1000']:
                        baseline_name = f'{dataset}_{baseline}{masked_str}_{metric}'
                        hypothesis_tests.append((PLS_name, baseline_name))

    start = time.time()
    print(f"Running bootstrap with {args.num_samples} samples")
    bootstrap_results(args)
    print(f"Run time: {(time.time() - start)/60:.2f} minutes")
