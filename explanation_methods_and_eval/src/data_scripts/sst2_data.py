import pandas
import jsonlines
import operator

def format_sst2(df, filename, maximum=None):
    dataset = []
    question = ""
    for index, row in df.iterrows():
        datapoint = {}
        datapoint['passage'] = row['sentence']
        datapoint['question'] = question
        if 'label' in row:
            datapoint['answer'] = row['label']
        num_words = len(row['sentence'].split(' '))
        dataset.append((num_words, datapoint))
        
    dataset.sort(key = operator.itemgetter(0))
    _, dataset = zip(*dataset)
    
    if maximum is not None:
        dataset = dataset[:maximum]
    
    with jsonlines.open(filename, mode='w') as writer:
        writer.write_all(dataset)

train = pandas.read_csv('data/sst2_raw/train.tsv', delimiter='\t')
dev = pandas.read_csv('data/sst2_raw/dev.tsv', delimiter='\t')
test = pandas.read_csv('data/sst2_raw/test.tsv', delimiter='\t')

format_sst2(train, 'data/sst2/train.jsonl')
format_sst2(dev, 'data/sst2/val.jsonl')
format_sst2(test, 'data/sst2/test.jsonl')

format_sst2(train, 'data/sst2_5000/train.jsonl', maximum=5000)
format_sst2(dev, 'data/sst2_5000/val.jsonl', maximum=5000)
format_sst2(test, 'data/sst2_5000/test.jsonl', maximum=5000)