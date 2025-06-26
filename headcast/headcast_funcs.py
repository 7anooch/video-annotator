import os
import re
import pandas as pd
import numpy as np

def extract_number(s):
    numbers = re.findall(r'\d+', s)  # Find all numbers in the string
    return int(numbers[-1])

def csv_path_list(path, invert=False, raw=False, outward_only=False, pc = True):
    csv_files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.csv')]
    file_list = [f for f in csv_files if 'mismatch' not in f]

    if raw:
        file_list = [f for f in file_list if 'raw' in f]
    else:
        file_list = [f for f in file_list if 'raw' not in f]

        if pc:
            file_list = [f for f in file_list if 'pc' in f]
        else:
            file_list = [f for f in file_list if 'pc' not in f]

        if invert:
            file_list = [f for f in file_list if 'invert' in f]
        else:
            file_list = [f for f in file_list if 'invert' not in f]

        if outward_only:
            file_list = [f for f in file_list if 'outward' in f]
        else:
            file_list = [f for f in file_list if 'outward' not in f]

    file_list.sort(key=extract_number)

    return file_list

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
            merged[special_case2] = 9  # Special case: set merged to 9
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


def transform_dict(dictionary, invert=False, outward_only=False):

    def extract_number(filename):
        base = os.path.basename(filename)
        parts = base.split('_')
        if len(parts) > 1 and parts[1].isdigit():
            return int(parts[1])
        return None

    new_dict = {}
    if invert:
        if outward_only:
            num_trials = np.sum([1 for key in dictionary.keys() 
                                if isinstance(key, str) and 'invert' in key and 'outward' in key])
        else:
            num_trials = np.sum([1 for key in dictionary.keys() 
                                if isinstance(key, str) and 'invert' in key])
    else:
        if outward_only:
            num_trials = np.sum([1 for key in dictionary.keys() 
                                if isinstance(key, str) and 'outward' in key and 'invert' not in key])
        else:
            num_trials = np.sum([1 for key in dictionary.keys() 
                            if isinstance(key, str) and 'invert' not in key and 'outward' not in key])
    

    print('num of trials: ', num_trials)
    for key in dictionary.keys():
        if isinstance(key, int):
            if len(list(dictionary[key].keys())) == 0:
                new_dict[key]['comparison'] = dictionary[key]
            else:
                skip_key = list(dictionary[key].keys())[0]
                print('skip key: ', skip_key)
                new_dict[key]['comparison'] = dictionary[key][skip_key]
        elif isinstance(key, str):
            num_trial = extract_number(key)
            if '.csv' in str(key):
                if num_trial not in new_dict:
                    new_dict[num_trial] = {}
                if 'Nitesh' in str(key):
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


def switch_left_right_labels(array):
    inverted = np.copy(array)
    inverted[array == 2] = -99  # temp for 2->5
    inverted[array == 5] = -98  # temp for 5->2
    inverted[array == 3] = -97  # temp for 3->6
    inverted[array == 6] = -96  # temp for 6->3

    inverted[inverted == -99] = 5
    inverted[inverted == -98] = 2
    inverted[inverted == -97] = 6
    inverted[inverted == -96] = 3
    return inverted

def get_labels(trial, outward_only=False, invert_left_right=True):
    columns = trial[:, [4, 5, 11, 12, 23, 21, 22]]
    # col 0: turn present (1/0)
    # col 1: turn direction (1/0/-1), 
    # col 2: cast present (1/0), 
    # col 3: cast direction (1/0/-1), 
    # col 4 accept/reject(1/0/-1)
    # col 5: cast outward/inward ( +1/0)
    # col 6: cast centerline ( +1/0)


    mode_data = np.zeros(columns.shape[0])
    mode_cast = np.zeros(columns.shape[0])
    mode_turn = np.zeros(columns.shape[0])
    mode_turn = np.zeros(columns.shape[0])
    mode_accept = np.zeros(columns.shape[0])
    mode_no_direction = np.zeros(columns.shape[0])

    if outward_only:
        centerline = np.logical_and(columns[:, 6] == 1, columns[:, 2] == 1)
        outward_cast = np.logical_and(columns[:, 5] == 1, centerline)

        right_cast = np.logical_and(columns[:, 3] == -1, outward_cast)
        left_cast = np.logical_and(columns[:, 3] == 1, outward_cast)
    else:
        right_cast = np.logical_and(columns[:, 3] == -1, columns[:, 2] == 1)
        left_cast = np.logical_and(columns[:, 3] == 1, columns[:, 2] == 1)

    right_turn = np.logical_and(columns[:, 1] == -1, columns[:, 0] == 1)
    left_turn = np.logical_and(columns[:, 1] == 1, columns[:, 0] == 1)

    if outward_only:
        cast_only = np.logical_and(columns[:, 0] == 0, outward_cast)
    else:
        cast_only = np.logical_and(columns[:, 0] == 0, columns[:, 2] == 1)

    turn_only = np.logical_and(columns[:, 0] == 1, columns[:, 2] == 0)

    accept = (columns[:, 4] == 1)
    reject = (columns[:, 4] == -1)

    mode_data[np.logical_and(right_cast, cast_only)] = 5
    mode_data[np.logical_and(left_cast, cast_only)] = 2
    mode_data[np.logical_and(right_turn, turn_only)] = 6
    mode_data[np.logical_and(left_turn, turn_only)] = 3

    mode_turn[right_turn] = 6
    mode_turn[left_turn] = 3

    mode_cast[right_cast] = 5
    mode_cast[left_cast] = 2

    mode_no_direction[np.logical_and(left_turn, turn_only)] = 3
    mode_no_direction[np.logical_and(right_turn, turn_only)] = 3

    mode_no_direction[np.logical_and(left_cast, cast_only)] = 2
    mode_no_direction[np.logical_and(right_cast, cast_only)] = 2

    mode_no_direction[np.logical_and(left_turn, left_cast)] = 4
    mode_no_direction[np.logical_and(right_turn, right_cast)] = 4
    mode_no_direction[np.logical_and(left_turn, right_cast)] = 4
    mode_no_direction[np.logical_and(right_turn, left_cast)] = 4

    mode_accept[accept] = 1
    mode_accept[reject] = -1

    if invert_left_right:
        mode_data = switch_left_right_labels(mode_data)
        mode_cast = switch_left_right_labels(mode_cast)
        mode_turn = switch_left_right_labels(mode_turn)

    return mode_data, mode_cast, mode_turn, mode_accept, mode_no_direction
