import torch
from torch_geometric.data import Data, Dataset
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import re
import matplotlib.pyplot as plt
from collections import defaultdict
from transformers import BertModel, BertTokenizer
from utils.HelperFunctions import get_embeddings_with_pos

def clean_graph(graph_str):
    #1:clean graph takes in a string, strips the white space aroung it, makes consistent indentations
    #2:changes all ': text' to start on its own line for the sake of graph making
    graph_str = graph_str.rstrip()

    indent = 4 #just pick an indentation level, will fix after everything is on a new line

    #1. consistent indentation- everything will be 4 spaces
    head_spaces = {}
    spacings = []
    all_lines = graph_str.splitlines()
    #start by retrieving all spaces
    for line in all_lines:
        dist = len(line) - len(line.lstrip())
        spacings.append(dist)
    # print(spacings)
    #Create new spacings
    no_dupes = []
    [no_dupes.append(x) for x in spacings if x not in no_dupes]
    # print(no_dupes)
    new_spacings = [sorted(no_dupes).index(x) * 4 for x in spacings]

    #switch out the spacings
    for line_i in range(len(all_lines)):
        all_lines[line_i]=new_spacings[line_i] *' ' + all_lines[line_i].lstrip()
    graph_str = '\n'.join(all_lines)

    #2: splits :x to be its own line
    lines= re.findall(r'.*:.*:.*',graph_str)# find lines with multiple ":"
    for line in lines:
        #actually does the replacement on new level
        line = line.split(':')
        for i in range(2,len(line)):
            num_spaces = len(line[0]) - len(line[0].lstrip())
            new_str = '\n' + ' ' * num_spaces + indent * ' ' + ":"
            graph_str = re.sub(r'(?<=[^\s]) :', new_str,graph_str,count = 1)

    return graph_str

def extract_node_2(node_str, i):
  node_str = re.sub(r'\"*_\(.*\)"', '"', node_str)
  node_str = node_str.strip()
  node_list = node_str.split('(')
  # print(node_str)
  if len(node_list) == 1:
    #it can either end or not depending on )
    if node_list[0][-1] == ")":
      # print("condition 1")
      edge = node_list[0].split(")")[0].split()[0].strip()
      g_id = node_list[0].split(")")[0].split()[1].strip()
      node = g_id

      return node, g_id, edge, 'remove', node_list[0].count(")")

    else:
      # print("condition 2", node_list[0])

      edge = node_list[0].split()[0].strip()
      g_id = node_list[0].split()[1].strip() if len(node_list[0].split()) > 1 else None
      node = g_id

      return node, g_id, edge, 'none', node_list[0].count(")")

  else:
    if node_list[0] == '':
      # print("condition 3")
      g_id = node_list[1].split("/")[0].strip()
      node = node_list[1].split("/")[1].strip()
      edge = None

      return node, g_id, edge, 'add', node_list[1].count(")") - 1

    elif node_list[1][-1] == ')':
      # print("condition 4")
      edge = node_list[0].strip()
      g_id = node_list[1].split(")")[0].split("/")[0].strip()
      node = node_list[1].split(")")[0].split("/")[1].strip()

      return node, g_id, edge, 'none', node_list[1].count(")") - 1
    else:
      # print("condition 5")
      g_id = node_list[1].split("/")[0].strip()
      node = node_list[1].split("/")[1].strip()
      edge = node_list[0].strip()

      return node, g_id, edge, 'add', node_list[1].count(")") - 1

def add_node_to_graph(index_to_node_dict, node_to_index_dict, g_id):
  if g_id not in node_to_index_dict.keys():
    node_to_index_dict[g_id] = len(node_to_index_dict)
    index_to_node_dict[len(index_to_node_dict)] = g_id

def create_graph(original_string, graph_str, var_mapping, target):
    # embeddings_with_pos = get_embeddings_with_pos(original_string)

    head_dict = {}

    parent_child_dict = defaultdict(list)
    node_to_index_dict = {}
    index_to_node_dict = {}
    index_to_embedding = {}
    is_leaf = {}

    # model_name = 'bert-base-uncased'

    # tokenizer = BertTokenizer.from_pretrained(model_name)

    found = 0
    train_mask_index = -1
    graph_str = clean_graph(graph_str)

    graph_stack = []

    for i in range(len(graph_str.splitlines())):
        #extact node and edge, get spacing to know what to attach it to
        curr_line = graph_str.splitlines()[i]

        node, g_id, edge, operation, num_remove = extract_node_2(curr_line,i)
        # print(node, g_id, edge, operation, num_remove)
        # print(graph_stack)
        if g_id == None or g_id == '':
          continue

        if g_id != node and g_id not in var_mapping.keys():
          var_mapping[g_id] = [0, 0]

        if operation == 'add':
          if edge == None:
            add_node_to_graph(index_to_node_dict, node_to_index_dict, g_id)
            graph_stack.append(g_id)

          else:
            if g_id in var_mapping.keys() or (edge == ':aspect' and node == 'blank' and graph_stack[-1] == target):
              add_node_to_graph(index_to_node_dict, node_to_index_dict, g_id)

              parent_child_dict[node_to_index_dict[graph_stack[-1]]].append(node_to_index_dict[g_id])
              graph_stack.append(g_id)

        elif operation == 'remove':
          if g_id in var_mapping.keys() or (edge == ':aspect' and node == 'blank' and graph_stack[-1] == target):
            add_node_to_graph(index_to_node_dict, node_to_index_dict, g_id)
            parent_child_dict[node_to_index_dict[graph_stack[-1]]].append(node_to_index_dict[g_id])

        else:
          if g_id in var_mapping.keys() or (edge == ':aspect' and node == 'blank' and graph_stack[-1] == target):
            add_node_to_graph(index_to_node_dict, node_to_index_dict, g_id)
            parent_child_dict[node_to_index_dict[graph_stack[-1]]].append(node_to_index_dict[g_id])

        if g_id in var_mapping.keys():
          left_end = var_mapping[g_id][0]
          right_end = var_mapping[g_id][1]

          if left_end == 0:
            # index_to_embedding[node_to_index_dict[g_id]] = torch.zeros(768)
            # index_to_embedding[node_to_index_dict[g_id]] = {"input_ids": torch.tensor([101, 103, 102]).unsqueeze(0), "attention_mask": torch.tensor([1]).unsqueeze(0)}
            # index_to_embedding[node_to_index_dict[g_id]] = tokenizer("[MASK]", truncation=True, return_tensors="pt", max_length=20, padding='max_length')
            index_to_embedding[node_to_index_dict[g_id]] = "[MASK]"
          else:
            # index_to_embedding[node_to_index_dict[g_id]] = torch.sum(embeddings_with_pos[left_end - 1:right_end], dim=0)
            # index_to_embedding[node_to_index_dict[g_id]] = tokenizer(original_string[left_end - 1:right_end], truncation=True, return_tensors="pt", max_length=20, padding='max_length')
            index_to_embedding[node_to_index_dict[g_id]] = original_string[left_end - 1:right_end]
          var_mapping[g_id] = [0, 0]

        if (edge == ':aspect' and node == 'blank' and graph_stack[-1] == target):
          # index_to_embedding[node_to_index_dict[g_id]] = torch.zeros(768)
          # index_to_embedding[node_to_index_dict[g_id]] = {"input_ids": torch.tensor([101, 103, 102]).unsqueeze(0), "attention_mask": torch.tensor([1]).unsqueeze(0)}
          # index_to_embedding[node_to_index_dict[g_id]] = tokenizer("[MASK]", truncation=True, return_tensors="pt", max_length=20, padding='max_length')
          index_to_embedding[node_to_index_dict[g_id]] = "[MASK]"
          train_mask_index = node_to_index_dict[g_id]

        for j in range(num_remove):
          graph_stack.pop()

    return parent_child_dict, node_to_index_dict, index_to_node_dict, index_to_embedding, train_mask_index

def create_data_graph(parent_child_dict, node_to_index_dict, index_to_node_dict, train_mask_index, embeddings, label):

  model_name = 'bert-base-uncased'

  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

  node_data = torch.tensor([])
  node_input_ids = torch.tensor([])
  node_attention_mask = torch.tensor([])
  # edges = [[0 for i in range(len(embeddings.keys()))] for i in range(len(embeddings.keys()))]
  edges = []
  train_mask = torch.zeros(len(embeddings))
  train_mask[train_mask_index] = 1

  for k in sorted(index_to_node_dict.keys()):
    temp = embeddings[k]
    node_tokens = torch.tensor([])
    # for child in parent_child_dict[k]:
    #   temp = temp + embeddings[child]
    #   edges.append([k, child])

    for child in parent_child_dict[k]:
      # print()
      # temp.append(embeddings[child])
      temp = temp + '[SEP]' + embeddings[child]
      edges.append([k, child])

    node_tokens = tokenizer(temp, truncation=True, return_tensors="pt", max_length=256, padding='max_length')
    node_input_ids = torch.cat((node_input_ids, node_tokens['input_ids']), dim=0)
    node_attention_mask = torch.cat((node_attention_mask, node_tokens['attention_mask']), dim=0)
    # print(node_tokens)

    # node_data = torch.cat((node_data, temp.unsqueeze(0)), dim=0)

  return Data(x=node_input_ids, edge_index=torch.tensor(edges).t().contiguous(), y=torch.tensor(label), train_mask=train_mask, input_ids=node_input_ids, attention_mask=node_attention_mask)