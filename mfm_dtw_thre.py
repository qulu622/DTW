# DTW-M_THRE

import pandas as pd # dataframe
import heapq as hq # heappush, heappop
import math # inf
import time # time
import os # listdir


def mfm_dtw_thre(ts1, ts2, minimum_distance): # Minimum First Method for DTW with threshold
    cell_dict = dict()

    ts1_index = 1
    ts2_index = 1

    ts1_len = len(ts1)
    ts2_len = len(ts2)

    cur_expand_value = abs(ts1[ts1_index] - ts2[ts2_index])


    unexpanded_cell_heap = [[math.inf, math.inf, math.inf]]

    while (ts1_len - 1, ts2_len - 1) not in cell_dict and (cur_expand_value < minimum_distance):
        if (ts1_index + 1, ts2_index + 1) not in cell_dict and (ts1_index + 1 < ts1_len) and (
                ts2_index + 1 < ts2_len):  # dia
            dia_cell = cur_expand_value + abs(ts1[ts1_index + 1] - ts2[ts2_index + 1])
            cell_dict[(ts1_index + 1, ts2_index + 1)] = dia_cell
            hq.heappush(unexpanded_cell_heap, [dia_cell, ts1_index + 1, ts2_index + 1])

        if (ts1_index + 1, ts2_index) not in cell_dict and (ts1_index + 1 < ts1_len):  # down
            down_cell = cur_expand_value + abs(ts1[ts1_index + 1] - ts2[ts2_index])
            cell_dict[(ts1_index + 1, ts2_index)] = down_cell
            hq.heappush(unexpanded_cell_heap, [down_cell, ts1_index + 1, ts2_index])

        if (ts1_index, ts2_index + 1) not in cell_dict and (ts2_index + 1 < ts2_len):  # right
            right_cell = cur_expand_value + abs(ts1[ts1_index] - ts2[ts2_index + 1])
            cell_dict[(ts1_index, ts2_index + 1)] = right_cell
            hq.heappush(unexpanded_cell_heap, [right_cell, ts1_index, ts2_index + 1])

        next_expand_cell = hq.heappop(unexpanded_cell_heap)
        # print(next_expand_cell)

        cur_expand_value = next_expand_cell[0]
        ts1_index = next_expand_cell[1]
        ts2_index = next_expand_cell[2]

    # print(cell_dict)

    if (ts1_len - 1, ts2_len - 1) in cell_dict:
        return cell_dict.get((ts1_len - 1, ts2_len - 1)), len(cell_dict)
    else:
        return math.inf, len(cell_dict)


def main():
    print('MFM_DTW_THRE')

    cur_dir = 'test'

    mfm_dtw_thre_data = pd.DataFrame(columns=['NAME', 'TIME', 'CELL', 'ACCURACY'])
    mfm_dtw_thre_data.to_csv('mfm_dtw_thre_' + cur_dir + '.csv', index=0)

    for file in os.listdir(cur_dir + '/1/'):
        print('Dataset {} begins.'.format(file))

        test_file = cur_dir + '/1/' + file + '/' + file + '_TEST.tsv'
        train_file = cur_dir + '/1/' + file + '/' + file + '_TRAIN.tsv'

        test_df = pd.read_csv(test_file, header=None, sep='\t')
        train_df = pd.read_csv(train_file, header=None, sep='\t')

        mfm_dtw_thre_total_time = 0
        mfm_dtw_thre_total_cell = 0
        mfm_dtw_thre_matched = 0
        mfm_dtw_thre_unmatched = 0

        for test_index, test_series in test_df.iterrows():
            minimum_distance = math.inf
            test_class = test_series.iat[0]
            for train_index, train_series in train_df.iterrows():
                mfm_dtw_thre_start = time.time()
                mfm_dtw_thre_distance, mfm_dtw_thre_cell = mfm_dtw_thre(test_series.to_list(), train_series.to_list(), minimum_distance)
                mfm_dtw_thre_end = time.time()
                mfm_dtw_thre_time = mfm_dtw_thre_end - mfm_dtw_thre_start

                mfm_dtw_thre_total_time += mfm_dtw_thre_time
                mfm_dtw_thre_total_cell += mfm_dtw_thre_cell

                if mfm_dtw_thre_distance < minimum_distance:
                    minimum_distance = mfm_dtw_thre_distance
                    train_class = train_series.iat[0]

            # print(minimum_distance)
            if test_class == train_class:
                mfm_dtw_thre_matched += 1
            else:
                mfm_dtw_thre_unmatched += 1

        # print(mfm_dtw_thre_total_time)
        # print(mfm_dtw_thre_total_cell)

        mfm_dtw_thre_data = mfm_dtw_thre_data.append({'NAME': file, 'TIME': str(mfm_dtw_thre_total_time), 'CELL': str(mfm_dtw_thre_total_cell),
                            'ACCURACY': str(mfm_dtw_thre_matched / (mfm_dtw_thre_matched + mfm_dtw_thre_unmatched))},
                           ignore_index=True)
        print(mfm_dtw_thre_data)
        mfm_dtw_thre_data.to_csv('mfm_dtw_thre_' + cur_dir + '.csv', mode='a', header=0, index=0)
        mfm_dtw_thre_data.drop(index=0, inplace=True)

    for file in os.listdir(cur_dir + '/10/'):
        print('Dataset {} begins.'.format(file))

        for i in range(1, 11):
            test_file = cur_dir + '/10/' + file + '/' + file + '_TEST' + str(i) + '.tsv'
            train_file = cur_dir + '/10/' + file + '/' + file + '_TRAIN' + str(i) + '.tsv'

            test_df = pd.read_csv(test_file, header=None, sep='\t')
            train_df = pd.read_csv(train_file, header=None, sep='\t')

            mfm_dtw_thre_total_time = 0
            mfm_dtw_thre_total_cell = 0
            mfm_dtw_thre_matched = 0
            mfm_dtw_thre_unmatched = 0

            for test_index, test_series in test_df.iterrows():
                minimum_distance = math.inf
                test_class = test_series.iat[0]
                for train_index, train_series in train_df.iterrows():
                    mfm_dtw_thre_start = time.time()
                    mfm_dtw_thre_distance, mfm_dtw_thre_cell = mfm_dtw_thre(test_series.to_list(), train_series.to_list(), minimum_distance)
                    mfm_dtw_thre_end = time.time()
                    mfm_dtw_thre_time = mfm_dtw_thre_end - mfm_dtw_thre_start

                    mfm_dtw_thre_total_time += mfm_dtw_thre_time
                    mfm_dtw_thre_total_cell += mfm_dtw_thre_cell

                    if mfm_dtw_thre_distance < minimum_distance:
                        minimum_distance = mfm_dtw_thre_distance
                        train_class = train_series.iat[0]

                # print(minimum_distance)
                if test_class == train_class:
                    mfm_dtw_thre_matched += 1
                else:
                    mfm_dtw_thre_unmatched += 1

            # print(mfm_dtw_thre_total_time)
            # print(mfm_dtw_thre_total_cell)

            mfm_dtw_thre_data = mfm_dtw_thre_data.append({'NAME': file, 'TIME': str(mfm_dtw_thre_total_time), 'CELL': str(mfm_dtw_thre_total_cell),
                                'ACCURACY': str(mfm_dtw_thre_matched / (mfm_dtw_thre_matched + mfm_dtw_thre_unmatched))},
                               ignore_index=True)
            print(mfm_dtw_thre_data)
            mfm_dtw_thre_data.to_csv('mfm_dtw_thre_' + cur_dir + '.csv', mode='a', header=0, index=0)
            mfm_dtw_thre_data.drop(index=0, inplace=True)


if __name__ == "__main__":
    main()
