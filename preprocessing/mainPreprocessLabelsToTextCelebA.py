import os
import sys
import random
import pandas as pd
import numpy as np


def make_text(row, attributes, max_len_text, random_shuffle=False, random_start=False):

    attr_str = [];
    for i, l in enumerate(row):
        if l == 1:
            attr_str.append(attributes[i-1]);

    if random_shuffle:
        random.shuffle(attr_str)
    labels_text = ', '.join(attr_str);
    while len(labels_text) > max_len_text:
        attr_str = attr_str[:-1];
        if random_shuffle:
            random.shuffle(attr_str)
        labels_text = ', '.join(attr_str);

    attr_text = labels_text.replace('_', ' ').lower()
    if random_start:
        if len(attr_text) < 255:
            start_index = np.random.randint(0, max_len_text - 1 - len(attr_text));
            attr_text = str('*' * start_index) + attr_text + str('*' * (max_len_text - len(attr_text) - start_index));
    return attr_text;


def attr_to_text(df, filename_out, max_length_text, shuffle_text=False, random_start_index=False):

    attr_names = df.columns.tolist();
    attr_names = attr_names[1:];
    print('attributes: ' + str(attr_names))

    df_text = pd.DataFrame(columns=['image_id', 'text']);
    df_text['image_id'] = df['image_id'];
    max_num_characters = 0;
    for index, row in df.iterrows():
        attr_str = make_text(row, attr_names, max_length_text, random_shuffle=shuffle_text, random_start=random_start_index)
        num_characters = len(attr_str);
        if num_characters > max_num_characters:
            max_num_characters = num_characters;
        else:
            if num_characters < max_length_text:
                attr_str = attr_str + str('*'*(max_length_text-num_characters));
                print(len(attr_str))
        print(attr_str)
        df_text.at[index, 'text'] = attr_str;

    print('df_text: ' + str(df_text.shape));
    print('max num characters: ' + str(max_num_characters))
    df_text.to_csv(filename_out, index=False)

if __name__ == '__main__':

    if len(sys.argv) < 3:
        print('data directory and stringlength are needed...exit')
        sys.exit()

    dir_data_in = sys.argv[1]
    dir_data_out = sys.argv[2]
    max_len_text = int(sys.argv[3]);

    # dir_data = '/home/' + uname + '/projects/multimodality/data/CelebA'
    filename_attr = os.path.join(dir_data_in, 'list_attr_celeba.csv')

    randomize_text_ordering = False
    randomize_start_index = True
    # max_len_text = 256;
    filename_text = os.path.join(dir_data_out, 'list_attr_text_' + str(max_len_text).zfill(3) + '_' + str(randomize_text_ordering) + '_' + str(randomize_start_index) + '_celeba.csv')

    df_attr = pd.read_csv(filename_attr);
    attr_to_text(df_attr, filename_text, max_len_text, shuffle_text=randomize_text_ordering, random_start_index=randomize_start_index);
