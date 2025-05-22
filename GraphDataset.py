import torch.nn.functional as F
import torch
import os
from torch_geometric.data import Data, Dataset, download_url
import jsonpickle
from Preprocess import create_graph_penman

class MyOwnDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return ['combined_data_annoted.json']

    @property
    def processed_file_names(self):
        # return os.listdir(self.processed_dir)
        return "no.pt"

    def download(self):
        pass

    def process(self):
        labelMapping = {
            "process": 0,
            "performance": 1,
            "endeavor": 2,
            "habitual": 3,
            "state": 4,
            "activity": 5,
            "none": 6
        }

        idx = 0

        for raw_path in self.raw_paths:

            with open(raw_path, 'r') as f:
                data = jsonpickle.decode(f.read())

                for i in range(len(data)):
                  # print("here", i)
                  # original_string, graph_str, var_mapping
                  # print("data", data[i])
                  # print("raw_path", data[i])
                  # graph = create_graph(data[i]['joined_sentence'], data[i]['graph'], data[i]['mapping'], data[i]['variable'])
                  # print("got graph")
                  labels = F.one_hot(torch.tensor(labelMapping[data[i]['aspect']] if data[i]['aspect'] in labelMapping.keys() else 0), num_classes=7)
                  # labels[labelMapping[data[i]['aspect']] if data[i]['aspect'] in labelMapping.keys() else 0] = 1

                  # 0 -> aspect
                  # 1 -> ended
                  # 2 -> complete

                  # labels[0:6] = F.one_hot(labelMapping[data[i]['aspect']] if data[i]['aspect'] in labelMapping.keys() else 0, num_classes=6)
                  labels = torch.cat((labels, torch.tensor([1 if data[i]['aspect'] == "performance" or data[i]['aspect'] == "endeavor" else 0])), dim=0)
                  labels = torch.cat((labels, torch.tensor([1 if data[i]['aspect'] == "performance" else 0])), dim=0)

                  # dataGraph = create_data_graph(graph[0], graph[1], graph[2], graph[4], graph[3], labels)
                  dataGraph = create_graph_penman(data[i]['graph'], data[i]['mapping'], data[i]['variable'], labels)

                  torch.save(dataGraph, os.path.join(self.processed_dir, f'data_{idx}.pt'))
                  idx += 1

    def len(self):
        list_dir = list(filter(lambda l: l.startswith("data_"), os.listdir(self.processed_dir)))

        return len(list_dir)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'), weights_only=False)
        return data

if __name__ == "__main__":
    dataset = MyOwnDataset(root="./UMRDataset")