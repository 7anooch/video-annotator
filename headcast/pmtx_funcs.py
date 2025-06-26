import pandas as pd
import numpy as np

# 0 instead of 1-indexed as in MATLAB
pmtx_map = {0: 'start_frame', 1: 'end_frame', 2: 'avg_speed', 3: 'is_run', 4: 'is_turn', 5: 'turn_direction', 
        6: 'peak_PAS_turn', 7: 'turn_peak_frame', 8: 'turn_start_frame', 9: 'turn_end_frame', 
        10: 'body_angle_turn_peak', 11: 'is_cast', 12: 'cast_direction', 13: 'peak_PAS_cast', 
        14: 'cast_peak_frame', 15: 'cast_start_frame', 16: 'cast_end_frame', 17: 'peak_head_angle', 
        18: 'single_stopcasts', 19: 'same_side', 20: 'number_stopcasts', 21: 'outward', 
        22: 'centerline', 23: 'accept', 24: 'turn_overlap'}

def read_pmtx(file):
    df = pd.read_csv(file, header=None)
    df = df.rename(columns = pmtx_map)
    return df

class PMTX:
    def __init__(self, file, invert = False):
        self.df = read_pmtx(file)
        self.pmtx_map = pmtx_map
        self.left_value = 1 if not invert else -1
        self.right_value = -1 if not invert else 1
        self.acceptance = self.df['accept'].values

        #cast, turn, both or neither
        self.cast = np.logical_and(            
            np.logical_and(self.df['is_cast'],  
                            self.df['outward']),
                            self.df['centerline'])
        self.turn = self.df['is_turn'].astype(bool)
        
        self.cast_turn = np.logical_and(self.turn, self.cast)
        self.straight = np.logical_and(~self.turn, ~self.cast)

        # cast/turn left/right
        self.left_turn = np.logical_and(self.turn, 
                                self.df['turn_direction'] == self.left_value)
        self.right_turn = np.logical_and(self.turn, 
                                self.df['turn_direction'] == self.right_value)

        self.left_cast = np.logical_and(self.cast, 
                                self.df['cast_direction'] == self.left_value)
        self.right_cast = np.logical_and(self.cast, 
                                self.df['cast_direction'] == self.right_value)

        # mutually exclusive cast/turn
        self.turn_only = np.logical_and(self.turn, ~self.cast)
        self.turn_only_left = np.logical_and(self.turn_only, 
                                self.df['turn_direction'] ==  self.left_value)
        self.turn_only_right = np.logical_and(self.turn_only, 
                                self.df['turn_direction'] == self.right_value)
        
        self.cast_only = np.logical_and(self.cast, ~self.turn)
        self.cast_only_left = np.logical_and(self.cast_only, 
                                self.df['cast_direction'] == self.left_value)
        self.cast_only_right = np.logical_and(self.cast_only, 
                                self.df['cast_direction'] == self.right_value)

        # cast/turn combinations with direction
        self.left_cast_turn = np.logical_and(self.left_cast, self.left_turn)
        self.right_cast_turn = np.logical_and(self.right_cast, self.right_turn)

        self.left_cast_right_turn = np.logical_and(self.left_cast, self.right_turn)
        self.right_cast_left_turn = np.logical_and(self.right_cast, self.left_turn)

    @property
    def cast_labels(self):
        labels = np.zeros(self.df.shape[0])
        labels[self.left_cast] = 2
        labels[self.right_cast] = 5
        return labels
    
    @property
    def turn_labels(self):
        labels = np.zeros(self.df.shape[0])
        labels[self.left_turn] = 3
        labels[self.right_turn] = 6
        return labels
    
    @property
    def mutually_ex_labels(self):
        "mutually exclusive cast/turn labels"
        labels = np.zeros(self.df.shape[0])
        labels[self.cast_only_left] = 2
        labels[self.turn_only_left] = 3
        labels[self.cast_only_right] = 5
        labels[self.turn_only_right] = 6
        return labels
    
    @property
    def merged_labels(self):
        "merged cast/turn labels"
        labels = np.zeros(self.df.shape[0])
        labels[self.cast_only_left] = 2
        labels[self.turn_only_left] = 3
        labels[self.cast_only_right] = 5
        labels[self.turn_only_right] = 6
        labels[self.left_cast_right_turn] = 7
        labels[self.right_cast_left_turn] = 8
        labels[self.left_cast_turn] = 9
        labels[self.right_cast_turn] = 11
        return labels
    
    @property
    def directionless_labels(self):
        "directionless cast/turn labels"
        labels = np.zeros(self.df.shape[0])
        labels[self.cast_only] = 2
        labels[self.turn_only] = 3
        labels[self.cast_turn] = 4
        return labels