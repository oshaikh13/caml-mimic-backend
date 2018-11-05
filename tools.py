"""
    Various utility methods
"""
import csv
import json
import math
import os
import pickle

import torch
from torch.autograd import Variable

import models
# from constants import *
import datasets
import numpy as np

def pick_model(args, dicts):
    """
        Use args to initialize the appropriate model
    """

    filter_size = int(args["filter_size"])
    args["embed_file"] = None
    model = models.ConvAttnPool(8921, args["embed_file"], filter_size, args["num_filter_maps"], 0, args["gpu"], dicts,
                                dropout=args["dropout"])
    
    if args["test_model"]:
        sd = torch.load(args["test_model"])
        model.load_state_dict(sd)
    if args["gpu"]:
        model.cuda()
    return model

def make_param_dict(args):
    """
        Make a list of parameters to save for future reference
    """
    param_vals = [args.Y, args.filter_size, args.dropout, args.num_filter_maps, args.rnn_dim, args.cell_type, args.rnn_layers, 
                  args.lmbda, args.command, args.weight_decay, args.version, args.data_path, args.vocab, args.embed_file, args.lr]
    param_names = ["Y", "filter_size", "dropout", "num_filter_maps", "rnn_dim", "cell_type", "rnn_layers", "lmbda", "command",
                   "weight_decay", "version", "data_path", "vocab", "embed_file", "lr"]
    params = {name:val for name, val in zip(param_names, param_vals) if val is not None}
    return params

def build_code_vecs(code_inds, dicts):
    """
        Get vocab-indexed arrays representing words in descriptions of each *unseen* label
    """
    code_inds = list(code_inds)
    ind2w, ind2c, dv_dict = dicts['ind2w'], dicts['ind2c'], dicts['dv']
    vecs = []
    for c in code_inds:
        code = ind2c[c]
        if code in dv_dict.keys():
            vecs.append(dv_dict[code])
        else:
            #vec is a single UNK if not in lookup
            vecs.append([len(ind2w) + 1])
    #pad everything
    vecs = datasets.pad_desc_vecs(vecs)
    return (torch.cuda.LongTensor(code_inds), vecs)

