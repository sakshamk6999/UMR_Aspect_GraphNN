from torch_geometric.nn import GCNConv, GATConv
import torch.nn.functional as F
import torch

class GCNModel(torch.nn.Module):
    def __init__(self, hidden_channels=512, MODEL_DIM=4096, num_hidden_layers=2):
        super().__init__()
        torch.manual_seed(1234567)
        # self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.linear1 = torch.nn.Linear(MODEL_DIM, MODEL_DIM)
        self.conv1 = GCNConv(MODEL_DIM, hidden_channels)

        self.conv_layers = [GCNConv(hidden_channels, hidden_channels) for _ in range(num_hidden_layers)]
        # self.conv2 = GCNConv(hidden_channels, hidden_channels)
        # self.conv3 = GCNConv(hidden_channels, hidden_channels)
        # self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.linear2 = torch.nn.Linear(hidden_channels, 9)

    def forward(self, x, edge_index, token_index):

        # print(x.shape)
        # print(input_ids.shape)
        x = self.linear1(x)
        # x = x.last_hidden_state[:,token_index,:]
        # print(x.shape)
        x = self.conv1(x, edge_index)
        # print(x.shape)

        for i in self.conv_layers:
            x = x.relu()
            x = i(x, edge_index)
        # x = x.relu()
        # print(x.shape)
        x = x[token_index]
        # print(x.shape)
        x = self.linear2(x)
        return x