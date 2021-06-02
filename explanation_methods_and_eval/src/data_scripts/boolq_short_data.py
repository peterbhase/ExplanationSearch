import jsonlines

def _select(input_filepath:str, output_path:str, maximum_length:int=200):
    train = []
    with jsonlines.open(input_filepath) as reader:
        for obj in reader:
            evidences = obj['evidences']
            if not (len(evidences) == 1 and len(evidences[0]) == 1):
                continue
            text = evidences[0][0]['text'].split(' ')
            if len(text) <= maximum_length:
                train.append(obj)
    print(len(train))

    with jsonlines.open(output_path, mode='w') as writer:
        writer.write_all(train)

_select('data/boolq/train.jsonl', 'data/boolq_short/train.jsonl', 200)
_select('data/boolq/val.jsonl', 'data/boolq_short/val.jsonl', 200)
_select('data/boolq/test.jsonl', 'data/boolq_short/test.jsonl', 200)