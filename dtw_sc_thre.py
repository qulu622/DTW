# SC_THRE

import numpy as np # matrix
import pandas as pd # dataframe
import math # inf
import time # time
import os # listdir


def dtw_sc_thre(ts1, ts2, minimum_distance): # DTW with SC-band
    cell = 0
    ts1_len = len(ts1) - 1
    ts2_len = len(ts2) - 1

    width = ts1_len * 0.1
    # print(width)

    dtw_matrix = np.zeros([ts1_len + 1, ts2_len + 1])
    for i in range(1, ts1_len + 1):
        dtw_matrix[i][0] = math.inf
    for j in range(1, ts2_len + 1):
        dtw_matrix[0][j] = math.inf

    for i in range(1, ts1_len + 1):
        for j in range(1, ts2_len + 1):
            dtw_matrix[i][j] = math.inf

    for i in range(1, ts1_len + 1):
        row_minimum_cell = math.inf
        for j in range(1, ts2_len + 1):
            if abs(i - j) < width:
                cell += 1
                dist = abs(ts1[i] - ts2[j])
                dtw_matrix[i][j] = dist + min(dtw_matrix[i - 1][j], dtw_matrix[i][j - 1], dtw_matrix[i - 1][j - 1])

                if dtw_matrix[i][j] < row_minimum_cell:
                    row_minimum_cell = dtw_matrix[i][j]

        if row_minimum_cell > minimum_distance:
            return math.inf, cell

    # print(dtw_matrix)

    return dtw_matrix[ts1_len][ts2_len], cell


def main():
    print('DTW_SC')

    cur_dir = 'test'

    dtw_sc_thre_data = pd.DataFrame(columns=['NAME', 'TIME', 'CELL', 'ACCURACY'])
    dtw_sc_thre_data.to_csv('dtw_sc_thre_' + cur_dir + '.csv', index=0)

    for file in os.listdir(cur_dir + '/1/'):
        print('Dataset {} begins.'.format(file))

        test_file = cur_dir + '/1/' + file + '/' + file + '_TEST.tsv'
        train_file = cur_dir + '/1/' + file + '/' + file + '_TRAIN.tsv'

        test_df = pd.read_csv(test_file, header=None, sep='\t')
        train_df = pd.read_csv(train_file, header=None, sep='\t')

        dtw_sc_thre_total_time = 0
        dtw_sc_thre_total_cell = 0
        dtw_sc_thre_matched = 0
        dtw_sc_thre_unmatched = 0

        for test_index, test_series in test_df.iterrows():
            minimum_distance = math.inf
            test_class = test_series.iat[0]
            for train_index, train_series in train_df.iterrows():
                dtw_sc_thre_start = time.time()
                dtw_sc_thre_distance, dtw_sc_thre_cell = dtw_sc_thre(test_series.to_list(), train_series.to_list(), minimum_distance)
                dtw_sc_thre_end = time.time()
                dtw_sc_thre_time = dtw_sc_thre_end - dtw_sc_thre_start

                dtw_sc_thre_total_time += dtw_sc_thre_time
                dtw_sc_thre_total_cell += dtw_sc_thre_cell

                if dtw_sc_thre_distance < minimum_distance:
                    minimum_distance = dtw_sc_thre_distance
                    train_class = train_series.iat[0]

            # print(minimum_distance)
            if test_class == train_class:
                dtw_sc_thre_matched += 1
            else:
                dtw_sc_thre_unmatched += 1

        # print(dtw_sc_thre_total_time)
        # print(dtw_sc_thre_total_cell)

        dtw_sc_thre_data = dtw_sc_thre_data.append({'NAME': file, 'TIME': str(dtw_sc_thre_total_time), 'CELL': str(dtw_sc_thre_total_cell),
                            'ACCURACY': str(dtw_sc_thre_matched/(dtw_sc_thre_matched+dtw_sc_thre_unmatched))}, ignore_index=True)
        print(dtw_sc_thre_data)
        dtw_sc_thre_data.to_csv('dtw_sc_thre_' + cur_dir + '.csv', mode='a', header=0, index=0)
        dtw_sc_thre_data.drop(index=0, inplace=True)

    for file in os.listdir(cur_dir + '/10/'):
        print('Dataset {} begins.'.format(file))

        for i in range(1, 11):
            test_file = cur_dir + '/10/' + file + '/' + file + '_TEST' + str(i) + '.tsv'
            train_file = cur_dir + '/10/' + file + '/' + file + '_TRAIN' + str(i) + '.tsv'

            test_df = pd.read_csv(test_file, header=None, sep='\t')
            train_df = pd.read_csv(train_file, header=None, sep='\t')

            dtw_sc_thre_total_time = 0
            dtw_sc_thre_total_cell = 0
            dtw_sc_thre_matched = 0
            dtw_sc_thre_unmatched = 0

            for test_index, test_series in test_df.iterrows():
                minimum_distance = math.inf
                test_class = test_series.iat[0]
                for train_index, train_series in train_df.iterrows():
                    dtw_sc_thre_start = time.time()
                    dtw_sc_thre_distance, dtw_sc_thre_cell = dtw_sc_thre(test_series.to_list(), train_series.to_list(), minimum_distance)
                    dtw_sc_thre_end = time.time()
                    dtw_sc_thre_time = dtw_sc_thre_end - dtw_sc_thre_start

                    dtw_sc_thre_total_time += dtw_sc_thre_time
                    dtw_sc_thre_total_cell += dtw_sc_thre_cell

                    if dtw_sc_thre_distance < minimum_distance:
                        minimum_distance = dtw_sc_thre_distance
                        train_class = train_series.iat[0]

                # print(minimum_distance)
                if test_class == train_class:
                    dtw_sc_thre_matched += 1
                else:
                    dtw_sc_thre_unmatched += 1

            # print(dtw_sc_thre_total_time)
            # print(dtw_sc_thre_total_cell)

            dtw_sc_thre_data = dtw_sc_thre_data.append({'NAME': file, 'TIME': str(dtw_sc_thre_total_time), 'CELL': str(dtw_sc_thre_total_cell),
                                'ACCURACY': str(dtw_sc_thre_matched / (dtw_sc_thre_matched + dtw_sc_thre_unmatched))}, ignore_index=True)
            print(dtw_sc_thre_data)
            dtw_sc_thre_data.to_csv('dtw_sc_thre_' + cur_dir + '.csv', mode='a', header=0, index=0)
            dtw_sc_thre_data.drop(index=0, inplace=True)


if __name__ == "__main__":
    main()
