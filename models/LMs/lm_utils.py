import torch
import os
import numpy as np
import time
import datetime
import pytz
import pandas as pd
import random

from utils.DataLoader import Data


class Data_for_LMs():

    def __init__(self, text, train_mask, val_mask, test_mask):
        """
        Data object to store the text information.
        """
        self.text = text
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask



def load_data(dataset_path, batch_size=9, val_ratio=0.15, test_ratio=0.15, use_llm=False, seed=0):
    num_classes = 2
    # for finetuning LM
    if use_llm:
        print('Using LLM')
    else:
        data, text = get_text_for_link_prediction(dataset_path, batch_size, val_ratio, test_ratio, seed=seed)

    return data, num_classes, text


def get_text_for_link_prediction(dataset_path, batch_size, val_ratio, test_ratio, seed=0):

    # Load data and train val test split
    graph_df = pd.read_csv(dataset_path + "/edge_list.csv")
    node_num = max(graph_df['u'].max(), graph_df['i'].max()) + 1
    rel_num = graph_df['r'].max() + 1
    if graph_df['label'].min() != 0:
        graph_df.label -= 1
    if 'GDELT' in dataset_path:
        graph_df.ts = graph_df.ts // 15
    cat_num = graph_df['label'].max() + 1

    # load entity text
    entity_text_reader = pd.read_csv(dataset_path + '/entity_text.csv', chunksize=10000)
    entity_text_list = []
    print("Loading entity embeddings...")
    for batch in entity_text_reader:
        id_batch = batch['i'].tolist()
        text_batch = batch['text'].tolist()
        if 0 in id_batch:
            id_batch = id_batch[1:]
            text_batch = text_batch[1:]
        if np.nan in text_batch:
            text_batch = ['NULL' if type(i) != str else i for i in text_batch]
        entity_text_list.extend(text_batch)

    # load edge text
    rel_text_reader = pd.read_csv(dataset_path + '/relation_text.csv', chunksize=10000)
    edge_text_list = []
    print("Loading relation embeddings...")
    for batch in rel_text_reader:
        id_batch = batch['i'].tolist()
        text_batch = batch['text'].tolist()
        if 0 in id_batch:
            id_batch = id_batch[1:]
            text_batch = text_batch[1:]
        if np.nan in text_batch:
            text_batch = ['NULL' if type(i) != str else i for i in text_batch]
        edge_text_list.extend(text_batch)


    src_node_ids = graph_df.u.values.astype(np.longlong)
    dst_node_ids = graph_df.i.values.astype(np.longlong)
    node_interact_times = graph_df.ts.values.astype(np.float64)
    edge_ids = graph_df.r.values.astype(np.longlong)
    labels = graph_df.label.values

    full_data = Data(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids, node_interact_times=node_interact_times,
                     edge_ids=edge_ids, labels=labels)

    # the setting of seed follows previous works
    random.seed(2020)

    full_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids, seed=0, unique_edge_num = len(edge_text_list))
    full_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(full_data.src_node_ids))), batch_size=batch_size, shuffle=False)

    text_for_prediction = []
    timestamp_for_mask = []
    for batch_idx, full_data_indices in enumerate(full_idx_data_loader):
        full_data_indices = full_data_indices.numpy()
        batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
            full_data.src_node_ids[full_data_indices], full_data.dst_node_ids[full_data_indices], \
                full_data.node_interact_times[full_data_indices], full_data.edge_ids[full_data_indices]

        _, batch_neg_dst_node_ids = full_neg_edge_sampler.sample(size=len(batch_src_node_ids))
        _, batch_neg_edge_ids = full_neg_edge_sampler.sample_edge_ids(size=len(batch_src_node_ids))

        batch_pos_text = [
            f"{node_interact_time}, {entity_text_list[src_id]}, {entity_text_list[dst_id]}, {edge_text_list[edge_id]}"
            for node_interact_time, src_id, dst_id, edge_id in zip(node_interact_times, batch_src_node_ids, batch_dst_node_ids, batch_edge_ids)
        ]
        batch_neg_text = [
            f"{node_interact_time}, {entity_text_list[src_id]}, {entity_text_list[dst_id]}, {edge_text_list[edge_id]}"
            for node_interact_time, src_id, dst_id, edge_id in zip(node_interact_times, batch_src_node_ids, batch_dst_node_ids, batch_neg_edge_ids)
        ]

        text_for_prediction.extend(batch_pos_text)
        text_for_prediction.extend(batch_neg_text)
        timestamp_for_mask.extend([node_interact_times, node_interact_times])

    # get the timestamp of validate and test set
    val_time, test_time = list(np.quantile(graph_df.ts, [(1 - val_ratio - test_ratio), (1 - test_ratio)]))

















    return data, text





# Create torch dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['node_id'] = idx
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])

        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


# * ============================= File Related =============================

def mkdir_p(path, log=True):
    """Create a directory for the specified path.
    Parameters
    ----------
    path : str
        Path name
    log : bool
        Whether to print result for directory creation
    """
    import errno
    if os.path.exists(path):
        return
    # print(path)
    # path = path.replace('\ ',' ')
    # print(path)
    try:
        os.makedirs(path)
        if log:
            print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path) and log:
            print('Directory {} already exists.'.format(path))
        else:
            raise


def get_dir_of_file(f_name):
    return os.path.dirname(f_name) + '/'

def init_path(dir_or_file):
    path = get_dir_of_file(dir_or_file)
    if not os.path.exists(path):
        mkdir_p(path)
    return dir_or_file


# * ============================= Time Related =============================


def time2str(t):
    if t > 86400:
        return '{:.2f}day'.format(t / 86400)
    if t > 3600:
        return '{:.2f}h'.format(t / 3600)
    elif t > 60:
        return '{:.2f}min'.format(t / 60)
    else:
        return '{:.2f}s'.format(t)


def get_cur_time(timezone='Asia/Shanghai', t_format='%m-%d %H:%M:%S'):
    return datetime.datetime.fromtimestamp(int(time.time()), pytz.timezone(timezone)).strftime(t_format)

def time_logger(func):
    def wrapper(*args, **kw):
        start_time = time.time()
        print(f'Start running {func.__name__} at {get_cur_time()}')
        ret = func(*args, **kw)
        print(
            f'Finished running {func.__name__} at {get_cur_time()}, running time = {time2str(time.time() - start_time)}.')
        return ret

    return wrapper
