import os
import re
import pandas as pd
import numpy as np

def extract_number(s):
    numbers = re.findall(r'\d+', s)  # Find all numbers in the string
    return int(numbers[-1])

def csv_path_list(path, invert=False):
    csv_files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.csv')]
    file_list = [f for f in csv_files if 'trial' in f]
    if invert:
        filt_file_list = [f for f in file_list if 'invert' in f and 'mismatch' not in f]
    else:
        filt_file_list = [f for f in file_list if 'invert' not in f and 'mismatch' not in f]
    #     file_list = [f for f in filt_csv_files if 'pc' in f and 'mismatch' not in f]
    # else:
    #     file_list = [f for f in filt_csv_files if 'trans' not in f and 'pc' not in f and 'mismatch' not in f]

    filt_file_list.sort(key=extract_number)

    return filt_file_list

def get_all_labels(csv_list, col = 2):
    if isinstance(csv_list, str):
        csv_list = [csv_list]
    all_annotations = []
    if isinstance(col, (list, tuple)) and len(col) == 2:
        for file in csv_list:
            df = pd.read_csv(file, header=None)
            col1 = df[col[0]].values
            col2 = df[col[1]].values
            merged =  np.zeros(len(col1))

            special_case = ((col1 == 2) & (col2 == 6)) | ((col1 == 6) & (col2 == 2))
            special_case2 = ((col1 == 3) & (col2 == 2)) | ((col1 == 2) & (col2 == 3))
            # print(len(special_case[special_case]))
            merged[special_case] = 7  # Special case: set merged to 7
            merged[special_case] = 9  # Special case: set merged to 9
            non_special = ~special_case & ~special_case2
            merged[non_special] = col1[non_special] + col2[non_special]

            all_annotations.append(merged)
        all_annotations = np.concatenate(all_annotations)
        return all_annotations
    else:
        for file in csv_list:
            df = pd.read_csv(file, header=None)
            all_annotations.append(df[col].values)
        all_annotations = np.concatenate(all_annotations)
    return all_annotations

def dir_data(path):
    firstcolmax = 0
    firstcolmin = 10
    secondcolmax = 0
    secondcolmin = 10

    for file in path:
        df = pd.read_csv(file, header=None)
        col1min = df[1].min()
        col1max = df[1].max()
        col2min = df[2].min()
        col2max = df[2].max()
        if col1min <= firstcolmin:
            firstcolmin = col1min
        if col1max >= firstcolmax:
            firstcolmax = col1max
        if col2min <= secondcolmin:
            secondcolmin = col2min
        if col2max >= secondcolmax:
            secondcolmax = col2max

        print('col1', np.unique(df[1].values))
        print('col2', np.unique(df[2].values))

    print(firstcolmin, firstcolmax)
    print(secondcolmin, secondcolmax)

def print_dict_tree(d, indent=0):
    for key, value in d.items():
        prefix = "    " * indent
        if isinstance(value, dict):
            print(f"{prefix}{key}/")
            print_dict_tree(value, indent + 1)
        else:
            try:
                length = value.shape
            except AttributeError:
                if isinstance(value, list):
                    length = f"list[{len(value)}]"
                else:
                    length = type(value).__name__
            print(f"{prefix}{key}: {length}")


def transform_dict(dictionary):

    def extract_number(filename):
        match = re.search(r'trial_(\d+)_pc(?:_invert)?\.csv', filename)
        if match:
            return int(match.group(1))
        return None

    new_dict = {}
    num_trials = np.sum([1 for key in dictionary.keys() if isinstance(key, str) and 'invert' in key])
    print('num of trials: ', num_trials)
    for key in dictionary.keys():
        if isinstance(key, int):
            skip_key = list(dictionary[key].keys())[0]
            new_dict[key]['comparison'] = dictionary[key][skip_key]
        elif isinstance(key, str):
            num_trial = extract_number(key)
            if '.csv' in str(key):
                if num_trial not in new_dict:
                    new_dict[num_trial] = {}
                if 'NAS' in str(key):
                    new_key = 'Nitesh'
                else:
                    new_key = 'Tanish'

                dict_for_key = {'seg_lengths': dictionary[key]['seg_lengths'],
                                'sequences': dictionary[key]['sequences'],
                                'lengths': dictionary[key]['lengths'],
                                'counts': dictionary[key]['counts']}
                new_dict[num_trial][new_key] = dict_for_key
                if 'accuracy' in dictionary[key]:
                    new_dict[num_trial]['prec/recall'] = dictionary[key]['prec/recall']
                    new_dict[num_trial]['accuracy']=  dictionary[key]['accuracy']
                    new_dict[num_trial]['conf_matrix']= dictionary[key]['conf_matrix']
                    new_dict[num_trial]['mismatch_frames']= dictionary[key]['mismatch_frames']
                    new_dict[num_trial]['mismatch_percent']= dictionary[key]['mismatch_percent']

    return new_dict


