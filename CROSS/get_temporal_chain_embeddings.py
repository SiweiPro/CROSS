import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
from transformers import AutoTokenizer, AutoConfig, AutoModel
import pickle


if __name__ == "__main__":
    precision = 8 # 四舍五入的小数位数
    version = 8
    batch_size = 100000
    pretrained_model_names = ['MiniLM']
    hidden_size = {'MiniLM': 384}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for pretrained_model_name in pretrained_model_names:
        # Prepare PLM modle and tokenizers
        config = AutoConfig.from_pretrained(pretrained_model_name)
        hidden_size = config.hidden_size

        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        PLM = AutoModel.from_pretrained(pretrained_model_name).to(device)
        print("PLM initialized")

        for data_set_name in ['GDELT', 'Enron', 'Googlemap_CT', 'ICEWS1819']:
            print(data_set_name)
            data_set_name = '../DyLink_Datasets/' + data_set_name
            node_raw_text = {}
            temporal_chain_raw_list = []
            temporal_chain_node_text = {}

            with open(data_set_name + f'/chain_results.json', 'r') as file:
                for line in file:
                    line = line.strip()
                    if line:
                        line_dict = json.loads(line)
                        line_dict['timestamp'] = int(line_dict['timestamp'])
                        line_dict['node_id'] = int(line_dict['node_id'])
                        temporal_chain_raw_list.append(line_dict)

            entity_text_reader = pd.read_csv(data_set_name + '/entity_text.csv', chunksize=batch_size)
            for batch_index, batch in enumerate(entity_text_reader):
                id_batch = batch['i'].tolist()
                text_batch = batch['text'].tolist()
                if 0 in id_batch:
                    id_batch = id_batch[1:]
                    text_batch = text_batch[1:]
                if np.nan in text_batch:
                    text_batch = ['NULL' if type(i) != str else i for i in text_batch]

                batch_dict = {id_: text for id_, text in zip(id_batch, text_batch)}
                node_raw_text.update(batch_dict)


            temporal_chain_raw_list = sorted(temporal_chain_raw_list, key=lambda x: x['timestamp'])

            for temporal_chain_data in temporal_chain_raw_list:
                node_id, timestamp, text = temporal_chain_data['node_id'], temporal_chain_data['timestamp'], temporal_chain_data['prediction']
                if text == '':
                    continue
                # print(node_id, timestamp, text)
                text = text.split('Summary: ', 1)[1].strip()
                if node_id not in temporal_chain_node_text.keys():
                    temporal_chain_node_text[node_id] = ([], [])
                    temporal_chain_node_text[node_id][0].append(-15.0)
                    temporal_chain_node_text[node_id][1].append(node_raw_text[node_id])
                temporal_chain_node_text[node_id][0].append(timestamp)
                temporal_chain_node_text[node_id][1].append(text)

            print(temporal_chain_raw_list[0])
            print(temporal_chain_node_text[116])

            num_node = len(temporal_chain_node_text)
            temporal_chain_node_embeddings = {}
            print("Loading temporal chain embeddings")
            for node_id, node_text_tuple in temporal_chain_node_text.items():
                timestamps, text_list = node_text_tuple[0], node_text_tuple[1]
                # For the nodes that first interacted with the target node, we insert a timestamp of -1.0, and the corresponding text embedding is a zero vector
                text_embeddings = []
                encoded_input = tokenizer(text_list, padding=True, truncation=True, max_length=512, return_tensors='pt')
                with torch.no_grad():
                    output = PLM(**encoded_input.to(device))[1]
                    output = output.cpu()
                for i in range(len(output)):
                    text_embeddings.append(np.around(output[i].numpy(), precision))
                assert len(text_embeddings) == len(timestamps)
                temporal_chain_node_embeddings[node_id] = (np.array(timestamps), np.array(text_embeddings))

            print([len(temporal_chain_node_embeddings), hidden_size])

            # A dict, with key: node_id, value: (timestamps: np.array, text_embeddings: np.array)
            pickle.dump(temporal_chain_node_embeddings, open(data_set_name + f'/temporal_chain_features.pkl', 'wb'))

            # np.save(data_set_name + f'/temporal_chain_features.pkl', temporal_chain_node_embeddings)
