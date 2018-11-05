"""
    Data loading methods
"""
from collections import defaultdict
import csv
import math
import numpy as np
import sys

# from constants import *

def pad_desc_vecs(desc_vecs):
    #pad all description vectors in a batch to have the same length
    desc_len = max([len(dv) for dv in desc_vecs])
    pad_vecs = []
    for vec in desc_vecs:
        if len(vec) < desc_len:
            vec.extend([0] * (desc_len - len(vec)))
        pad_vecs.append(vec)
    return pad_vecs


def load_vocab_dict(args, vocab_file):
    #reads vocab_file into two lookups (word:ind) and (ind:word)
    vocab = set()
    with open(vocab_file, 'r') as vocabfile:
        for i,line in enumerate(vocabfile):
            line = line.rstrip()
            if line != '':
                vocab.add(line.strip())
    #hack because the vocabs were created differently for these models
    if args['public_model'] and args['Y'] == 'full' and args['model'] == 'conv_attn':
        ind2w = {i:w for i,w in enumerate(sorted(vocab))}
    else:
        ind2w = {i+1:w for i,w in enumerate(sorted(vocab))}
    w2ind = {w:i for i,w in ind2w.items()}
    return ind2w, w2ind

def load_lookups(args, desc_embed=False):
    """
        Inputs:
            args: Input arguments
            desc_embed: true if using DR-CAML
        Outputs:
            vocab lookups, ICD code lookups, description lookup, description one-hot vector lookup
    """
    #get vocab lookups
    with open("./datafiles/codes.txt") as f:
        codesFile = f.readlines()


    codes = set()

    for code in codesFile:
        codes.add(code.strip())
    
    ind2c = {i:c for i,c in enumerate(sorted(codes))}
    c2ind = {c:i for i,c in ind2c.items()}

    ind2w, w2ind = load_vocab_dict(args, args['vocab'])

    dicts = {'ind2w': ind2w, 'w2ind': w2ind, 'c2ind': c2ind, 'ind2c': ind2c}
    return dicts


def load_code_descriptions(version="mimic3"):
    #load description lookup from the appropriate data files
    desc_dict = defaultdict(str)
    with open("./datafiles/D_ICD_DIAGNOSES.csv", 'r') as descfile:
        r = csv.reader(descfile)
        #header
        next(r)
        for row in r:
            code = row[1]
            desc = row[-1]
            desc_dict[reformat(code, True)] = desc
    with open("./datafiles/D_ICD_PROCEDURES.csv", 'r') as descfile:
        r = csv.reader(descfile)
        #header
        next(r)
        for row in r:
            code = row[1]
            desc = row[-1]
            if code not in desc_dict.keys():
                desc_dict[reformat(code, False)] = desc
    with open('./datafiles/ICD9_descriptions.txt', 'r') as labelfile:
        for i,row in enumerate(labelfile):
            row = row.rstrip().split()
            code = row[0]
            if code not in desc_dict.keys():
                desc_dict[code] = ' '.join(row[1:])
    return desc_dict

def reformat(code, is_diag):
    """
        Put a period in the right place because the MIMIC-3 data files exclude them.
        Generally, procedure codes have dots after the first two digits, 
        while diagnosis codes have dots after the first three digits.
    """
    code = ''.join(code.split('.'))
    if is_diag:
        if code.startswith('E'):
            if len(code) > 4:
                code = code[:4] + '.' + code[4:]
        else:
            if len(code) > 3:
                code = code[:3] + '.' + code[3:]
    else:
        code = code[:2] + '.' + code[2:]
    return code

