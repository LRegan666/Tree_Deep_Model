import os
import time
import random
import multiprocessing as mp
import pandas as pd
import numpy as np
from construct_tree import TreeInitialize
# import traceback


LOAD_DIR = os.path.dirname(os.path.abspath(__file__)) + '/datasets/UserBehavior_sp.csv'


def _time_window_stamp():
    boundaries = ['2017-11-26 00:00:00', '2017-11-27 00:00:00', '2017-11-28 00:00:00',
                  '2017-11-29 00:00:00', '2017-11-30 00:00:00', '2017-12-01 00:00:00',
                  '2017-12-02 00:00:00', '2017-12-03 00:00:00', '2017-12-04 00:00:00']
    for i in range(len(boundaries)):
        time_array = time.strptime(boundaries[i], "%Y-%m-%d %H:%M:%S")
        time_stamp = int(time.mktime(time_array))
        boundaries[i] = time_stamp
    return boundaries


def _time_converter(x, boundaries):
    tag = -1
    if x > boundaries[-1]:
        tag = 9
    else:
        for i in range(len(boundaries)):
            if x <= boundaries[i]:
                tag = i
                break
    return tag


def _mask_padding(data, max_len):
    size = data.shape[0]
    raw = data.values
    mask = np.array([[-2] * max_len for _ in range(size)])
    for i in range(size):
        mask[i, :len(raw[i])] = raw[i]
    return mask.tolist()


def data_process():
    data_raw = pd.read_csv(LOAD_DIR, header=None,
                           names=['user_ID', 'item_ID', 'category_ID', 'behavior_type', 'timestamp'])
    data_raw = data_raw[:10000]
    user_list = data_raw.user_ID.drop_duplicates().to_list()
    user_dict = dict(zip(user_list, range(len(user_list))))
    data_raw['user_ID'] = data_raw.user_ID.apply(lambda x: user_dict[x])
    item_list = data_raw.item_ID.drop_duplicates().to_list()
    item_dict = dict(zip(item_list, range(len(item_list))))
    data_raw['item_ID'] = data_raw.item_ID.apply(lambda x: item_dict[x])
    category_list = data_raw.category_ID.drop_duplicates().to_list()
    category_dict = dict(zip(category_list, range(len(category_list))))
    data_raw['category_ID'] = data_raw.category_ID.apply(lambda x: category_dict[x])
    behavior_dict = dict(zip(['pv', 'buy', 'cart', 'fav'], range(4)))
    data_raw['behavior_type'] = data_raw.behavior_type.apply(lambda x: behavior_dict[x])
    time_window = _time_window_stamp()
    data_raw['timestamp'] = data_raw.timestamp.apply(_time_converter, args=(time_window,))
    random_tree = TreeInitialize(data_raw)
    _ = random_tree.random_binary_tree()
    data = data_raw.groupby(['user_ID', 'timestamp'])['item_ID'].apply(list).reset_index()
    data['behaviors'] = data_raw.groupby(['user_ID',
                                         'timestamp'])['behavior_type'].apply(list).reset_index()['behavior_type']
    data['behavior_num'] = data.behaviors.apply(lambda x: len(x))
    mask_length = data.behavior_num.max()
    data = data[data.behavior_num >= 10]
    data = data.drop(columns=['behavior_num'])
    data['item_ID'] = _mask_padding(data['item_ID'], mask_length)
    data['behaviors'] = _mask_padding(data['behaviors'], mask_length)
    data_train, data_validate, data_test = data[:-200], data[-200:-100], data[-100:]
    cache = (user_dict, item_dict, behavior_dict, random_tree)
    return data_train, data_validate.reset_index(drop=True), data_test.reset_index(drop=True), cache


def _single_node_sample(item_id, node, root):
    sample_num = 200
    samples = []
    positive_info = {}
    i = 0
    while node:
        if node.item_id is None:
            single_sample = [item_id, node.val, 0, [1, 0]]
        else:
            single_sample = [item_id, node.item_id, 1, [1, 0]]
        samples.append(single_sample)
        positive_info[i] = node
        node = node.parent
        i += 1
    j, k = i-1, 0
    level_nodes = [root]
    while level_nodes:
        tmp = []
        for node in level_nodes:
            if node.left:
                tmp.append(node.left)
            if node.right:
                tmp.append(node.right)
        if j >= 0:
            level_nodes.remove(positive_info[j])
        if level_nodes:
            if len(level_nodes) <= 2*k:
                index_list = range(len(level_nodes))
                sample_num -= len(level_nodes)
            else:
                index_list = random.sample(range(len(level_nodes)), 2*k)
                sample_num -= 2*k
            if j == 0:
                index_list = random.sample(range(len(level_nodes)), sample_num + 2*k)
            for level_index in index_list:
                if level_nodes[level_index].item_id is None:
                    single_sample = [item_id, level_nodes[level_index].val, 0, [0, 1]]
                else:
                    single_sample = [item_id, level_nodes[level_index].item_id, 1, [0, 1]]
                samples.append(single_sample)
        level_nodes = tmp
        k += 1
        j -= 1
    samples = pd.DataFrame(samples, columns=['item_ID', 'node', 'is_leaf', 'label'])
    return samples


def _tree_generate_worker(task_queue, sample_queue):
    while True:
        try:
            item_id, node, root = task_queue.get()
            node_sample = _single_node_sample(item_id, node, root)
            sample_queue.put(node_sample)
        except Exception as err:
            print("Tree Worker Process Exception Info: {}".format(str(err)))
        finally:
            task_queue.task_done()


def tree_generate_samples(items, leaf_dict, root):
    jobs = mp.JoinableQueue()
    tree_samples = mp.Queue()
    for _ in range(8):
        process = mp.Process(target=_tree_generate_worker, args=(jobs, tree_samples))
        process.daemon = True
        process.start()
    total_samples = None
    for i in range(0, len(items), 50):
        sub_items = items[i:i+50]
        for item in sub_items:
            jobs.put((item, leaf_dict[item], root))
        jobs.join()
        batch_samples = []
        while not tree_samples.empty():
            tree_sample = tree_samples.get_nowait()
            batch_samples.append(tree_sample)
        if total_samples is None:
            total_samples = pd.concat(batch_samples, ignore_index=True)
        else:
            batch_samples = pd.concat(batch_samples, ignore_index=True)
            total_samples = pd.concat([total_samples, batch_samples], ignore_index=True)
    return total_samples


def _single_data_merge(data, tree_data):
    complete_data = None
    item_ids = np.array(data.item_ID)
    item_ids = item_ids[item_ids != -2]
    for item in item_ids:
        samples_tree_item = tree_data[tree_data.item_ID == item][['node', 'is_leaf', 'label']].reset_index(drop=True)
        size = samples_tree_item.shape[0]
        data_extend = pd.concat([data] * size, axis=1, ignore_index=True).T
        data_item = pd.concat([data_extend, samples_tree_item], axis=1)
        if complete_data is None:
            complete_data = data_item
        else:
            complete_data = pd.concat([complete_data, data_item], axis=0, ignore_index=True)
    return complete_data


def _merge_generate_worker(tree_data, task_queue, sample_queue):
    while True:
        try:
            data_row = task_queue.get()
            complete_sample = _single_data_merge(data_row, tree_data)
            sample_queue.put(complete_sample)
        except Exception as err:
            print("Merge Worker Process Exception Info: {}".format(str(err)))
            # traceback.print_exc()
        finally:
            task_queue.task_done()


def merge_samples(data, tree_sample):
    jobs = mp.JoinableQueue()
    complete_samples = mp.Queue()
    for _ in range(8):
        process = mp.Process(target=_merge_generate_worker, args=(tree_sample, jobs, complete_samples))
        process.daemon = True
        process.start()
    data_complete = None
    train_size = data.shape[0]
    for i in range(0, train_size, 50):
        for _ in range(50):
            if i == train_size:
                break
            jobs.put(data.iloc[i])
            i += 1
        jobs.join()
        batch_samples = []
        while not complete_samples.empty():
            single_data_sample = complete_samples.get_nowait()
            batch_samples.append(single_data_sample)
        if data_complete is None:
            data_complete = pd.concat(batch_samples, ignore_index=True)
        else:
            batch_samples = pd.concat(batch_samples, ignore_index=True)
            data_complete = pd.concat([data_complete, batch_samples], ignore_index=True)
    return data_complete


class Dataset(object):
    def __init__(self, data, batch_size, shuffle=False):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        self.data = self.data.drop(columns=['user_ID', 'timestamp'])
        N, B = self.data.shape[0], self.batch_size
        idxs = np.arange(N)
        if self.shuffle:
            np.random.shuffle(idxs)
        if self.data.shape[1] > 2:
            return ((np.array(self.data.loc[idxs[i:i+B], 'item_ID'].tolist()),
                     self.data.loc[idxs[i:i+B], 'node'].values[:, None],
                     self.data.loc[idxs[i:i+B], 'is_leaf'].values[:, None],
                     np.array(self.data.loc[idxs[i:i+B], 'label'].tolist())) for i in range(0, N, B))
        else:
            return (np.array(self.data.loc[idxs[i:i+B], 'item_ID'].tolist()) for i in range(0, N, B))


if __name__ == '__main__':
    data_train, data_validate, data_test, cache = data_process()
    user_dict, item_dict, _, tree = cache
    items = tree.items
    total_samples = tree_generate_samples(items, tree.leaf_dict, tree.root)
    data_complete = merge_samples(data_train, total_samples)
    dtrain = Dataset(data_complete, 50, shuffle=True)
    for item, node, is_leaf, label in dtrain:
        print(item[:5])
        print('===========================================================')
        print(node[:5])
        print('===========================================================')
        print(is_leaf[:5])
        print('===========================================================')
        print(label[:5])
        break

