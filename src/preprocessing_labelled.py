#!/usr/bin/env python3

import os
import json
import sys

input_dir = './main/resources/data/train/'
output_file = './main/resources/data.json'
input_dir_pos = input_dir+'pos/'
input_dir_neg = input_dir+'neg/'

pos_files = os.listdir(input_dir_pos)
neg_files = os.listdir(input_dir_neg)

for file in pos_files:
    file_tokens = file.split('_')
    r_id = file_tokens[0]
    stars = file_tokens[1].split('.')[0]
    label = 1
    with open(input_dir_pos+file, 'r') as file:
        review = file.read()
        json_obj = {
            'id': r_id,
            'stars': stars,
            'label': label,
            'review': review,
        }
        with open(output_file, 'a') as o_file:
            o_file.write(json.dumps(json_obj, ensure_ascii=False)+'\n')

for file in neg_files:
    file_tokens = file.split('_')
    r_id = file_tokens[0]
    stars = file_tokens[1].split('.')[0]
    label = 0
    with open(input_dir_neg+file, 'r') as file:
        review = file.read()
        json_obj = {
            'id': r_id,
            'stars': stars,
            'label': label,
            'review': review,
        }
        with open(output_file, 'a') as o_file:
            o_file.write(json.dumps(json_obj, ensure_ascii=False)+'\n')
