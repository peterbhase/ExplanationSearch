import jsonlines

def _convert(input_filepath:str, output_path:str):
    with jsonlines.open(input_filepath) as reader:
        for dictionary in reader:
            for key, value in dictionary.items():
                dictionary[key] = [[0.0, v] for v in value]

    with jsonlines.open(output_path, mode='w') as writer:
        writer.write(dictionary)

_convert('outputs/lime/boolq_raw_val_saliency_scores_1d.txt', 'outputs/lime/boolq_raw_val_saliency_scores.txt')