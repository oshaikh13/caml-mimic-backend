import datasets
import tools
import csv
import numpy as np
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
import torch

def evaluate():
    """
        Load data, build model, create optimizer, create vars to hold metrics, etc.
    """

    args = {
        "data_path": "train_full.csv",
        "vocab": "./datafiles/vocab.csv",
        "model": "conv_attn",
        "filter_size": 10,
        "num_filter_maps": 50,
        "dropout": .2,
        "lr": .0001,
        "gpu": False,
        "test_model": "model.pth",
        "public_model": "true",
        "Y": "full",
        "n_epochs": 200
    }

    dicts = datasets.load_lookups(args)

    model = tools.pick_model(args, dicts)

    with open('./datafiles/example_note.txt', 'r') as notefile:
        note = notefile.read()

        tokens = [t.lower() for t in tokenizer.tokenize(note) if not t.isnumeric()]
        text = '"' + ' '.join(tokens) + '"'

        w2ind = dicts["w2ind"]
        text = [int(w2ind[w]) if w in w2ind else len(w2ind)+1 for w in text.split()]
        test(model, False, text, dicts) #testing
    
def test(model, gpu, text, dicts):
    """
        Testing loop.
        Returns metrics
    """

    ind2w, w2ind, ind2c, c2ind = dicts['ind2w'], dicts['w2ind'], dicts['ind2c'], dicts['c2ind']

    model.eval()

    data = torch.tensor([text])
    # data.unsqueeze(1)
    print(data.size())
    if gpu:
        data = data.cuda()
        target = target.cuda()
    model.zero_grad()

    #get an attention sample for 2% of batches
    get_attn = True
    output, loss, alpha = model(data, get_attention=get_attn)

    output = output.data.cpu().numpy()[0]
    alpha = alpha.data.cpu().numpy()[0]

    # print(output)
    max_8 = np.argpartition(output, -8)[-8:]

    max_word_indexes = {}
    for code in max_8: max_word_indexes[ind2c[code]] = np.argmax(alpha[code])

    print(max_word_indexes)
    

evaluate()