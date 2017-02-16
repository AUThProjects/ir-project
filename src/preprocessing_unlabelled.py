#!/usr/bin/env python3

import os
import json

input_dir = './src/main/resources/data/test/'
output_file = './src/main/resources/data/data_unlabelled.json'

files = os.listdir(input_dir)

for file in files:
    file_tokens = file.split('_')
    r_id = file_tokens[0]
    label = 0
    with open(input_dir+file, 'r') as file:
        review = file.read()
        json_obj = {
            'id': r_id,
            'review': review,
        }
        with open(output_file, 'a') as o_file:
            o_file.write(json.dumps(json_obj, ensure_ascii=False)+'\n')
