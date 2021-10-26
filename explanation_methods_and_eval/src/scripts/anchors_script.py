import torch
from overrides import overrides
import os
import time
import math
import sys
import numpy as np
import random
from anchor import anchor_text
import spacy

from allennlp.data import PyTorchDataLoader
from allennlp.common.tqdm import Tqdm
from allennlp.nn import util as nn_util

from src.scripts.base_script import BaseScript

class AnchorScript(BaseScript):

    @overrides
    def init_parser_args(self, parser):
        super().init_parser_args(parser)
        parser.add_argument('--batch_size', default=10, type=int, help='note adjusting this does not actually change internal Anchors batch size')
        parser.add_argument('--num_samples', type=int)
        parser.add_argument('--eval_only', action='store_true')

    @overrides
    def init_base_dir(self, args):
        if not args.debug:
            self.base_dir = f'outputs/anchors/{args.dataset}/{args.datasplit}'
        else:
            self.base_dir = 'outputs/anchors/debug'
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

    def anchor_saliency(self, args, task_model, batch, sparsity):
        output_labels = task_model.vocab.get_token_to_index_vocabulary('labels').keys()
        anchor_explainer = anchor_text.AnchorText(class_names=output_labels, use_unk_distribution=True, mask_string='UNKWORDZ')

        with torch.no_grad():
            full_output_dict = task_model._forward(**batch)

        batch_document, batch_predicted_label = \
            self.stack_input(batch['document'], full_output_dict['predicted_label'], args.batch_size)
        doc_tokens = batch['metadata'][0]['doc_tokens']
        doc_length = len(doc_tokens)

        def predict_proba(text_list):
            binary_masks = [[1 if t != 'UNKWORDZ' else 0 for t in tokens.split(' ')] for tokens in text_list]
            preds = []
            for i in range(0, len(binary_masks), args.batch_size):
                true_batch_size = min(len(binary_masks) - i, args.batch_size)
                for j in range(true_batch_size):
                    batch_document['task']['mask'][j, ..., 1:(doc_length+1)] = torch.tensor(binary_masks[i+j])
                with torch.no_grad():
                    output_dict = task_model._forward(document=batch_document, label=batch_predicted_label)
                batch_preds = output_dict['probs'][:true_batch_size].cpu().data.numpy()
                batch_preds = np.argmax(batch_preds, axis=-1)
                preds.extend(batch_preds)

            assert len(preds) == len(text_list)
            return np.array(preds)

        # have to strip doc tokens of transformers tokenizer artifacts, so that anchors processes it correctly
        input_text = ' '.join([str(x) for x in doc_tokens])
        num_features = math.ceil(sparsity * doc_length)
        full_predicted_label = full_output_dict['predicted_label'].item()
        explanation = anchor_explainer.explain_instance(
            input_text,
            predict_proba,
            max_num_samples=args.num_samples,
            max_anchor_size=num_features,
            verbose=False,
        )

        # get locations of anchor words, turn into binary vector
        input_list = input_text.split()
        saliency = [0.0 for i in range(doc_length)]
        for idx in range(len(input_list)):
            if input_list[idx] in explanation.names():
                saliency[idx] = 1.0

        return np.array(saliency)

    @overrides
    def run(self):
        args = self.initialize(enable_warnings=False)

        reader = self.init_task_reader(args)
        data, vocab = self.read_data(args, reader)
        data_loader = PyTorchDataLoader(data, batch_size=1, shuffle=False)
        task_model = self.init_task_model(args, vocab)
        task_model = self.load_task_model(args, task_model)

        if args.eval_only:
            masks_path = os.path.join(self.base_dir, f'{args.exp_name}_best_masks.json')
            assert os.path.exists(masks_path)
            saliency_dict = json.load(open(masks_path))
        else:
            saliency_dict = {}

        # fix the Anchors seed! do not let vary with args.seed
        np.random.seed(0)
        random.seed(0)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        best_mask_dict = {}        

        with torch.no_grad():
            task_model.eval()
            iterator = iter(data_loader)
            generator_tqdm = Tqdm.tqdm(iterator)            

            start = time.time()
            for idx, batch in enumerate(generator_tqdm):
                batch = nn_util.move_to_device(batch, args.cuda_device)
                instance_id = batch['metadata'][0]['instance_id']
                best_mask_dict[instance_id] = {}

                for objective in ['suff', 'comp']:
                    for sparsity in self.sparsity_list:
                        use_sparsity = sparsity if objective == 'suff' else 1-sparsity

                        # Get Anchor explanation
                        if args.eval_only:
                            explanation = [saliency_dict[instance_id][sparsity][objective]]
                        else:
                            explanation = self.anchor_saliency(args, task_model, batch, use_sparsity)
                            explanation = [explanation.tolist()]
                        
                        # update metrics
                        min_suff, max_comp = self.update_search_metrics(
                            args, task_model, batch, sparsity, explanation, objective=objective)
                        if objective == 'suff':
                            best_mask_dict[instance_id][sparsity] = {'suff': min_suff.best_doc_mask}
                        if objective == 'comp':
                            best_mask_dict[instance_id][sparsity].update({'comp': max_comp.best_doc_mask})

                result_dict, result_str = self.format_metrics()
                generator_tqdm.set_description(result_str, refresh=True)

        end = time.time()
        result_dict['time'] = end - start
        result_dict['args'] = str(sys.argv)
        print('-----Final Result-----')
        for k, v in result_dict.items():
            print(f"  {k} : {v}")
        print('----------')

        prefix = args.exp_name
        self.write_json_to_file(result_dict, self.base_dir, 'final_result.json', prefix)
        self.write_json_to_file(best_mask_dict, self.base_dir, 'best_masks.json', prefix)
        self.write_metrics_to_csv(self.base_dir, 'per_point_metrics.csv', prefix)


if __name__=='__main__':
    script = AnchorScript()
    script.run()