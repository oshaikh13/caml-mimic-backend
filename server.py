import datasets
import tools
import csv
import numpy as np
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
import torch

from flask_cors import CORS
from flask import Flask, jsonify, request
app = Flask(__name__)
CORS(app)

def setup():
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
    dicts["code_descs"] = datasets.load_code_descriptions()
    model = tools.pick_model(args, dicts)

    # with open('./datafiles/example_note.txt', 'r') as notefile:
    #     note = notefile.read()
    #     test(model, False, note, dicts) #testing

    @app.route("/", methods=['POST'])
    def hello():
        note = request.get_json()["note"]
        print(note)
        if (note is None): return "Note not found"
        results = test(model, False, note, dicts)
        return jsonify(results)
    
def test(model, gpu, note, dicts):
    """
        Testing loop.
        Returns metrics
    """

    tokens = [t.lower() for t in tokenizer.tokenize(note) if not t.isnumeric()]
    text = '"' + ' '.join(tokens) + '"'
    w2ind = dicts["w2ind"]

    original = text.split()
    text = [int(w2ind[w]) if w in w2ind else len(w2ind)+1 for w in text.split()]

    ind2w, w2ind, ind2c, c2ind = dicts['ind2w'], dicts['w2ind'], dicts['ind2c'], dicts['c2ind']

    model.eval()

    data = torch.tensor([text])
    if gpu:
        data = data.cuda()
        target = target.cuda()
    model.zero_grad()

    #get an attention sample for 2% of batches
    get_attn = True
    output, loss, alpha = model(data, get_attention=get_attn)

    output = output.data.cpu().numpy()[0]
    alpha = alpha.data.cpu().numpy()[0]

    max_8 = np.argpartition(output, -8)[-8:]

    max_code_word_indexes = {}
    max_code_word_descs = {}

    for code in max_8: max_code_word_indexes[ind2c[code]] = int(np.argmax(alpha[code]) - 1)
    for code in max_8: max_code_word_descs[ind2c[code]] = dicts["code_descs"][ind2c[code]]
    
    # print(max_code_word_indexes)
    # print(max_code_word_descs)
    
    predictionLabels = []

    for code in max_8:
        prediction = {}
        location = int(np.argmax(alpha[code]) - 1)
        prediction["code"] = ind2c[code]
        prediction["codeDescription"] = dicts["code_descs"][ind2c[code]]
        prediction["kgram"] = original[location - 2 : location + 2]
        predictionLabels.append(prediction)

    return predictionLabels
setup()