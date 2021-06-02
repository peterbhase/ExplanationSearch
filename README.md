# Search Methods for Sufficient, Socially-Aligned Feature Importance Explanations with In-Distribution Counterfactuals

This is the repository for the paper [here](arxiv-link), including code for measuring OOD-ness of explanation counterfactuals and several new search methods for identifying explanations.

First we give a demo of Parallel Local Search, and then we describe steps for replicating experimental results.

# Requirements

Experiment scripts in the folders here have different requirements. After installing PyTorch, use the `requirements.txt` files in each for the corresponding experiments. For the demo below, you can first run `pip install -r OOD_analysis/requirements.txt`.

## Parallel Local Search Demo

Parallel Local Search is the best explanation method we test, as per the Sufficiency and Comprehensiveness metrics. We give a demo for an NLI model, using SNLI dev points. Note that the model used in this demo is not a Counterfactual-Trained model, which is what we ultimately recommend using when creating feature importance explanations. 

The demo script `pls_demo.py` can be run on a GPU with device ID 0 with the following command:

`python pls_demo.py --device 0 --num_search 1000 --objective suff --explanation_sparsity .2`

where `--num_search` specifies the compute budget in terms of forward passes, `--objective` specifies either the Sufficiency (`suff`) or Comprehensivess (`comp`) explanation objectives, and `--explanation_sparsity` determines the proportion of tokens kept in the explanation. On an NVIDIA RTX 2080, the output of this script is:

```
-------------------------------------
NOTE: the current Replace(x,e) function is the Attention Mask function
This function uses the binary explanation e as the attention mask over tokens
Blanks (like this: __) in the output of this script represent 0s in the input attention mask
Actual words in the script output are seen by the model
-------------------------------------
Loading model...
Searching for explanation for point 0...took 2.23 seconds!
Model input:  <s>A dog swims in a pool.</s></s>A puppy is swiming.</s>
Explanation:   __ __ swim __ __ __ __ __ __ puppy __ swim __ __
Model predicts neutral with pred prob: 0.937 | Pred prob for explanation: 0.969 | suff: -3.11 points
Input label: neutral

Searching for explanation for point 1...took 5.33 seconds!
Model input:  <s>A man on a beach chopping coconuts with machete.</s></s>A man is outdoors</s>
Explanation:   __ man __ __ beach __ __ __ __ __ __ __ __ __A man __ __
Model predicts entailment with pred prob: 0.972 | Pred prob for explanation: 0.978 | suff: -0.59 points
Input label: entailment

Searching for explanation for point 2...took 5.20 seconds!
Model input:  <s>Young people sit on the rocks beside their recreational vehicle.</s></s>A group of senior citizens sit by their RV.</s>
Explanation:  Young people __ __ __ __ __ __ __ __ __ __ __ of senior citizens __ __ __ __ __
Model predicts contradiction with pred prob: 0.916 | Pred prob for explanation: 1.000 | suff: -8.34 points
Input label: contradiction

Searching for explanation for point 3...took 5.55 seconds!
Model input:  <s>After playing with her other toys, the baby decides that the guitar seems fun to play with as well.</s></s>A blonde baby.</s>
Explanation:  After __ __ __ other __, __ __ decides __ __ __ __ __ __ __ __ __ __ __ __ blonde __ __
Model predicts neutral with pred prob: 0.994 | Pred prob for explanation: 0.998 | suff: -0.43 points
Input label: neutral

Searching for explanation for point 4...took 5.52 seconds!
Model input:  <s>A teenage girl in winter clothes slides down a decline in a red sled.</s></s>A girl is by herself on a sled.</s>
Explanation:   __ teenage girl __ __ __ slides __ __ __ __ __ __ __ __A girl __ __ __ __ __ __ __
Model predicts neutral with pred prob: 0.844 | Pred prob for explanation: 0.982 | suff: -13.81 points
Input label: entailment

```

Try with `--objective comp --explanation_sparsity .8` to see what tokens are removed in order to decrease the model confidence in its original prediction. 

## Counterfactual OOD-ness Experiments

Before running any experiments, set the default `--save_dir` and `--cache_dir` arguments in `main.py`. All experiments are managed via the `run_tasks.py` script. See the local variables `data`, `model`, `masking_style`, `sparsity`, and `source_sparsity` for experimental conditions.

To train 10 Standard models on device `GPU`, run

`python run_tasks.py --gpu GPU --experiment baseline --seeds 10`

To train 10 Counterfactual-Trained models, run

`python run_tasks.py --gpu GPU --experiment joint_baseline --seeds 10`

See training reports for these models in `training_reports`

We save the predictions on ablated inputs, i.e. inputs given by Replace(x,e), in the `result_sheets` folder by running

`python run_tasks.py --gpu GPU --experiment masked --seeds 10`

and

`python run_tasks.py --gpu GPU --experiment joint_masked --seeds 10`

Plots of the results are then made within the R markdown file, `make_plots/analysis_OOD.Rmd`. Move the data from `result_sheets` here or edit the filepaths appropriately.

## Explanation Experiments

`src/scripts` includes scripts for:

- Training models, including with Counterfactual Training
- Explanation experiments, for the following methods:
    - LIME
    - Vanilla Gradient
    - Integrated Gradients
    - Random Search
    - Exhaustive Search
    - Gradient Search
    - Taylor Search
    - Ordered Search
    - Parallel Local Search

Experiments are run across seeds, datasets, and Standard/Counterfactual-Trained models with `run_tasks.py`.

First, task models must be trained. To train 10 models, run

`python run_tasks.py --gpu GPU --experiment task_model --seeds 10 `

The experiments condition is controlled by local variables `model`, `data`, and `masking` (0 for Standard, 1 for Counterfactual-Trained).

For evaluating the explanations methods, you can for example run the experiment with a budget of 1000 forward passes for Parallel Local Search with

`python run_tasks.py --gpu GPU --experiment parallel_local_search --seeds 10 `

See scripts for the other methods in `src/scripts`.

To compute CIs and hypohesis tests for all of the evaluation results, run

`python compute_CIs.py`

which will save the results in `data`. See the local variables `methods`, `datasets`, and `model` for controlling what settings are tested. For plotting the search trajectories over time, move the `*_all_trajectories.csv` files from `outputs` to `make_plots` and use `analysis_search.Rmd`.