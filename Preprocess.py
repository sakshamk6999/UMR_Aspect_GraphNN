import torch
from torch_geometric.data import Data, Dataset
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import re
import matplotlib.pyplot as plt
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel
from utils.HelperFunctions import get_embeddings_with_pos
from sklearn.model_selection import train_test_split
import jsonpickle
import penman
# from GraphDataset import MyOwnDataset


MODEL_NAME = 'google-bert/bert-large-uncased'
MODEL_DIM = 1024

def split_data(df, target_column, train_size=0.7, random_state=None):
    train_df, test_df = train_test_split(
        df, train_size=train_size, stratify=df[target_column], random_state=random_state
    )
    return train_df, test_df

def get_embeddings_with_pos(text):
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
  model = AutoModel.from_pretrained(MODEL_NAME)

  embeddings = []

  num_tokens = []

  for word in text.split():
    num_tokens.append(len(tokenizer.tokenize(word)))
    
  tokenized = tokenizer(text, truncation=True, return_tensors="pt")
  embedded_input = model(**tokenized)

  curr = 1
  for i in range(len(num_tokens)):
    embeddings.append(torch.sum(embedded_input.last_hidden_state[0][curr:curr+num_tokens[i]], dim=0))
    curr += num_tokens[i]

  embeddings = torch.stack(embeddings)

  return embeddings

def create_graph_penman(graph_str, var_mapping, target, label):
  embeddings_with_pos = get_embeddings_with_pos(graph_str)

  penman_graph = penman.decode(graph_str)

  instances = penman_graph.instances()
  edges = penman_graph.edges()

  node_data = torch.tensor([])

  node2index = {}
  index2node = {}
  curr_index = 0

  edges_rec = []

  target_index = -1
  # if g_id in var_mapping.keys():

  for i in range(len(instances)):
    s = instances[i].source

    if s not in var_mapping.keys():
      var_mapping[s] = [[0, 0]]
    indices = var_mapping[s]
    # print("got indices", indices)
    temp = torch.zeros(embeddings_with_pos.shape[-1])

    for j in indices:
      if s not in node2index.keys():
        node2index[s] = curr_index
        index2node[curr_index] = s
        curr_index += 1

      if j[0] != 0:
        temp += torch.sum(embeddings_with_pos[j[0] - 1:j[0]], dim=0)

    node_data = torch.cat((node_data, temp.unsqueeze(0)), dim=0)

    if s == target:
      target_index = node2index[s]
      node_data = torch.cat((node_data, torch.zeros(embeddings_with_pos.shape[-1]).unsqueeze(0)), dim=0)
      node2index['aspect'] = curr_index
      index2node[curr_index] = 'aspect'
      curr_index += 1


  for edge in edges:
    source = edge.source
    target = edge.target
    edges_rec.append([node2index[source], node2index[target]])

  edges_rec.append([target_index, node2index['aspect']])
  edges_data = torch.tensor(edges_rec).t().contiguous()

  return Data(x=node_data, edge_index=edges_data, y=torch.tensor(label), target_node=torch.tensor(node2index['aspect']))

def createAlignedDataset(dataDir):
  doc_aligned_with_joined_sentence = open(dataDir + "/combined_data_annoted.json", 'w')

  with open(dataDir + "/combined_all.json", 'r') as f:
      data = jsonpickle.decode(f.read())
      print("data is", f.read())
      for i in range(len(data)):
        data[i]['joined_sentence'] = ' '.join(data[i]['sentence'].values())
        data[i]['mapping'] = {}

        for item in data[i]['alignment']:
          key = item.split(":")[0]
          items = item.split(":")[1].strip().split(',')
          mappings = []

          for item in items:
            item = item.split("-")
            mappings.append([int(item[0]), int(item[-1])])

          data[i]['mapping'][key] = mappings

      doc_aligned_with_joined_sentence.write(jsonpickle.encode(data))

  doc_aligned_with_joined_sentence.close()

if __name__ == "__main__":
  dataDir = "./UMR Data"
  # createAlignedDataset(dataDir)
  # dataset = MyOwnDataset(root=dataDir)