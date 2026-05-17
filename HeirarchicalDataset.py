import torch.nn.functional as F
import torch
import os
from torch_geometric.data import Data, Dataset, download_url, InMemoryDataset
from Preprocess import create_graph_penman, create_graph_object
import json
import pandas as pd
from tqdm import tqdm
from conf import child_to_parent, label_to_node, node_to_label, parent_to_child

class HeirarchicalDataset(Dataset):
    def __init__(self, root, split='train', **kwargs):
        self.split = split
        print("the split is", self.split)
        super().__init__(root, **kwargs)

    @property
    def raw_file_names(self):
        return ['train_data.json', 'combinedFinalized.csv', 'alignedEmbeddings.jsonl', 'splits.json']

    @property
    def processed_file_names(self):
        # return "no.pt"
        return "data_test_0.pt"

    def download(self):
        pass

    def process(self):
        idx = 0

        train_data = self.raw_paths[0]
        data_frame_path = self.raw_paths[1]
        aligned_emb_path = self.raw_paths[2]
        split_keys_path = self.raw_paths[3]

        data_list = []

        with open(aligned_emb_path, "r") as f:
            for line in f:
                try:
                    json_object = json.loads(line)
                    data_list.append(json_object)
                except json.JSONDecodeError:
                    # Handle cases where a line might be empty or malformed
                    continue
        
        with open(train_data, "r") as f:
            train_data = json.load(f)
        
        total_df = pd.read_csv(data_frame_path)

        with open(split_keys_path, 'r') as f:
            split_keys_path = json.load(f)
        
        keys_in_interest = split_keys_path[self.split]

        with tqdm(total=len(keys_in_interest)) as pbar:
            for d in keys_in_interest:

                df_id = d

                emb = None

                for i in data_list:
                    if list(i.keys())[0] == df_id:
                        emb = i
                        break

                df_for_id = total_df[total_df['id'] == df_id]

                for i in range(len(df_for_id)):
                    df = df_for_id.iloc[i]

                    graph_str = df['graph']
                    target_variable = df['variable_name']
                    aspect_label = df['adjudicated']
                    mapping_string = df['alignment']
                    
                    labels = torch.zeros(1, 11)
                    actual_label = label_to_node['none']

                    if aspect_label == "performance":
                        cols_to_fill = [label_to_node['performance'], label_to_node['perfective'], label_to_node['process'], label_to_node['aspect']]
                        labels[:,cols_to_fill] = 1
                        actual_label = label_to_node['performance']
                    elif aspect_label == "endeavor":
                        cols_to_fill = [label_to_node['endeavor'], label_to_node['perfective'], label_to_node['atelic'], label_to_node['process'], label_to_node['imperfective'], label_to_node['aspect']]
                        labels[:,cols_to_fill] = 1
                        actual_label = label_to_node['endeavor']
                    elif aspect_label == "activity":
                        cols_to_fill = [label_to_node['activity'], label_to_node['atelic'], label_to_node['process'], label_to_node['imperfective'], label_to_node['aspect']]
                        labels[:,cols_to_fill] = 1
                        actual_label = label_to_node['activity']
                    elif aspect_label == "process":
                        cols_to_fill = [label_to_node['process'], label_to_node['aspect']]
                        labels[:,cols_to_fill] = 1
                        actual_label = label_to_node['process']
                    elif aspect_label == "state":
                        cols_to_fill = [label_to_node['state'], label_to_node['imperfective'], label_to_node['aspect']]
                        labels[:,cols_to_fill] = 1
                        actual_label = label_to_node['state']
                    elif aspect_label == "habitual":
                        cols_to_fill = [label_to_node['habitual'], label_to_node['aspect']]
                        labels[:,cols_to_fill] = 1
                        actual_label = label_to_node['habitual']
                    else:
                        cols_to_fill = [label_to_node['none']]
                        labels[:,cols_to_fill] = 1

                    # print("aspect label", aspect_label)
                    # print("labels", labels)

                    try:
                        dataGraph = create_graph_object(graph_str, target_variable, mapping_string, torch.tensor(emb[df_id]), labels, torch.tensor(actual_label))
                    except Exception as e:
                        print("df_id", df_id)
                        print("Exception", e)
                        # raise Exception("error")
                        continue

                    torch.save(dataGraph, os.path.join(self.processed_dir, f'data_{self.split}_{idx}.pt'))
                    idx += 1
                pbar.update(1)
                #     if idx == 100:
                #         break

                # if idx == 100:
                #     break

        print("all done !")
        # for raw_path in self.raw_paths:

        #     with open(raw_path, 'r') as f:
        #         data = jsonpickle.decode(f.read())

        #         for i in range(len(data)):
        #           # print("here", i)
        #           # original_string, graph_str, var_mapping
        #           # print("data", data[i])
        #           # print("raw_path", data[i])
        #           # graph = create_graph(data[i]['joined_sentence'], data[i]['graph'], data[i]['mapping'], data[i]['variable'])
        #           # print("got graph")
        #           labels = F.one_hot(torch.tensor(labelMapping[data[i]['aspect']] if data[i]['aspect'] in labelMapping.keys() else 0), num_classes=7)
        #           # labels[labelMapping[data[i]['aspect']] if data[i]['aspect'] in labelMapping.keys() else 0] = 1

        #           # 0 -> aspect
        #           # 1 -> ended
        #           # 2 -> complete

        #           # labels[0:6] = F.one_hot(labelMapping[data[i]['aspect']] if data[i]['aspect'] in labelMapping.keys() else 0, num_classes=6)
        #           labels = torch.cat((labels, torch.tensor([1 if data[i]['aspect'] == "performance" or data[i]['aspect'] == "endeavor" else 0])), dim=0)
        #           labels = torch.cat((labels, torch.tensor([1 if data[i]['aspect'] == "performance" else 0])), dim=0)

        #           # dataGraph = create_data_graph(graph[0], graph[1], graph[2], graph[4], graph[3], labels)
        #           dataGraph = create_graph_penman(data[i]['graph'], data[i]['mapping'], data[i]['variable'], labels)

        #           torch.save(dataGraph, os.path.join(self.processed_dir, f'data_{idx}.pt'))
        #           idx += 1

    def len(self):
        list_dir = list(filter(lambda l: l.startswith(f"data_{self.split}"), os.listdir(self.processed_dir)))

        return len(list_dir)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'data_{self.split}_{idx}.pt'), weights_only=False)
        return data
    
if __name__ == "__main__":
    dataset = HeirarchicalDataset(root="./UMRDataset")