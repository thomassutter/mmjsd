import sys,os
import numpy as np
import pandas as pd

import torch
from torchvision.utils import save_image

from utils.constants_svhnmnist import indices
from utils.text import tensor_to_text

def append_list_to_list_linear(l1, l2):
    for k in range(0, len(l2)):
        if isinstance(l2[k], str):
            l1.append(l2[k]);
        else:
            l1.append(l2[k].item());
    return l1;

def write_samples_text_to_file(samples, filename):
    file_samples = open(filename, 'w');
    for k in range(0, len(samples)):
        file_samples.write(''.join(samples[k]) + '\n');
    file_samples.close();

def getText(samples):
    lines = []
    for k in range(0, len(samples)):
        lines.append(''.join(samples[k])[::-1])
    text = '\n\n'.join(lines)
    print(text)
    return text

def write_samples_img_to_file(samples, filename, img_per_row=1):
    save_image(samples.data.cpu(), filename, nrow=img_per_row);


def save_generated_samples_singlegroup(flags, batch_id, alphabet, group_name, samples):
    if group_name == 'real':
        dir_name = flags.dir_gen_eval_fid_real;
    elif group_name == 'random_sampling':
        dir_name = flags.dir_gen_eval_fid_random;
    elif group_name.startswith('dynamic_prior'):
        mod_store = flags.dir_gen_eval_fid_dynamicprior;
        dir_name = os.path.join(mod_store, '_'.join(group_name.split('_')[-2:]));
    elif group_name.startswith('cond_gen_1a2m'):
        mod_store = flags.dir_gen_eval_fid_cond_gen_1a2m;
        dir_name = os.path.join(mod_store, group_name.split('_')[-1]);
    elif group_name.startswith('cond_gen_2a1m'):
        mod_store = flags.dir_gen_eval_fid_cond_gen_2a1m;
        dir_name = os.path.join(mod_store, '_'.join(group_name.split('_')[-2:]));
    elif group_name == 'cond_gen':
        dir_name = flags.dir_gen_eval_fid_cond_gen;
    else:
        print('group name not defined....exit')
        sys.exit();

    for k, key in enumerate(samples.keys()):
        dir_f = os.path.join(dir_name, key);
        if not os.path.exists(dir_f):
            os.makedirs(dir_f);

    cnt_samples = batch_id * flags.batch_size;
    for k in range(0, flags.batch_size):
        for i, key in enumerate(samples.keys()):
            f_out = os.path.join(dir_name,  key, str(cnt_samples).zfill(6) + '.png')
            if key.startswith('img'):
                save_image(samples[key][k], f_out, nrow=1);
            elif key == 'text':
                write_samples_text_to_file(tensor_to_text(alphabet, samples[key][k].unsqueeze(0)), f_out);
        cnt_samples += 1;
