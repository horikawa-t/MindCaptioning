#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Utility functions'''

__author__ = 'horikawa-t'

# Imports
import os
import subprocess
import glob
import random
import torch
import numpy as np
from math import factorial
from itertools import permutations

# Function to get directory names
def getDN(path):
    files = sorted(glob.glob(path))
    return files

# Function to get file names
def getFN(path):
    files = sorted([filename.split('/')[-1] for filename in glob.glob(path)])
    return files

# Function to generate random integers
def randsample(min, max, cnt, sortflag=False, revflag=False):
    list = []
    i = 0
    while cnt != i:
        r = random.randint(min, max)
        if r not in list:
            list.append(r)
            i += 1
    if sortflag:
        list.sort(reverse=revflag)
    return list

# Function to extract labels corresponding to filenames
def extract_labels(all_labels, all_filenames, query_filenames):
    extracted_labels = []
    for query_fname in query_filenames:
        for fname, label in zip(all_filenames, all_labels):
            if query_fname == fname:
                extracted_labels.append(label)
                break  # Stop searching once a match is found
    return extracted_labels

# Function to check and set directory
def setdir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
            try:
                os.chmod(path, 0o777)
            except:
                print('Cannot change the permission of the directory')
        except:
            print('Crash: failed to make directories or it already exists.')
    return path

# Function to fix seed
def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# Function to compute number of permutations
def nPr(n, r):
    return int(factorial(n) / factorial(n - r))

# Function to generate random index
def generate_random_idx(n, nshuffle_base, remove_original=1):
    possible_permutations = nPr(n, n)

    # revise nshuffle
    nshuffle = min(nshuffle_base, possible_permutations - remove_original)

    shuffled_idx = set()
    while len(shuffled_idx) < nshuffle and possible_permutations:
        shuffled_ind = random.sample(range(n), k=n)

        if remove_original and shuffled_ind == list(range(n)):
            continue  # Skip adding the original sentence

        shuffled_idx.add(tuple(shuffled_ind))

    return shuffled_idx

