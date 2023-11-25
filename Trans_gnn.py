import torch 
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing , GATv2Conv, GATConv,SGConv, GCNConv


# Trans_GCN
class Trans_GCN(torch.nn.Module):
    def __init__(self, dataset,hid_dim):
        super(Trans_GCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, hid_dim)
        self.conv2 = GCNConv(hid_dim, dataset.num_classes )
  


    def forward_once(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        return x
    def forward(self, data, Trans_data):
        y = self.forward_once(data)
        z = self.forward_once(Trans_data)





        return F.log_softmax(y, dim=1), (1-F.cosine_similarity(y,z)), F.log_softmax(z, dim=1), F.log_softmax(y,dim=1), F.log_softmax(y, dim=1)




#////////////////////////////////////////////////////////////////////////////////////////////////////////
# Trans_GAT

class Trans_GAT(torch.nn.Module):
    def __init__(self, dataset,hid_dim,heads):
        super(Trans_GAT, self).__init__()
        self.conv1 = GATConv(dataset.num_node_features, hid_dim, heads, dropout=0.6)
        # On the Pubmed dataset, use `heads` output heads in `conv2`.
        self.conv2 = GATConv(hid_dim * heads, dataset.num_classes, heads=1,
                             concat=False, dropout=0.6)
      


    def forward_once(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x
    def forward(self, data, Trans_data):
        y = self.forward_once(data)
        z = self.forward_once(Trans_data)





        return F.log_softmax(y, dim=1), (1-F.cosine_similarity(y,z)), F.log_softmax(z, dim=1), F.log_softmax(y,dim=1), F.log_softmax(y, dim=1)




#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# Trans_GATv2

class Trans_GATV2(torch.nn.Module):
    def __init__(self, dataset,hid_dim,heads):
        super(Trans_GATV2, self).__init__()
        self.conv1 = GATv2Conv(dataset.num_node_features, hid_dim, heads, dropout=0.6)
        # On the Pubmed dataset, use `heads` output heads in `conv2`.
        self.conv2 = GATv2Conv(hid_dim * heads, dataset.num_classes, heads=1,
                             concat=False, dropout=0.6)
  


    def forward_once(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=0.3, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        return x
    def forward(self, data, Trans_data):
        y = self.forward_once(data)
        z = self.forward_once(Trans_data)





        return F.log_softmax(y, dim=1), (1-F.cosine_similarity(y,z)), F.log_softmax(z, dim=1), F.log_softmax(y,dim=1), F.log_softmax(y, dim=1)




#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# Trans_SGN

class Trans_SGC(torch.nn.Module):
    def __init__(self, dataset,hid_dim):
        super(Trans_SGC, self).__init__()
        self.conv1 = SGConv(dataset.num_node_features, hid_dim,k=1)
        self.conv2 = SGConv(hid_dim, dataset.num_classes, k=1 )



    def forward_once(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        return x
    def forward(self, data, Trans_data):
        y = self.forward_once(data)
        z = self.forward_once(Trans_data)





        return F.log_softmax(y, dim=1), (1-F.cosine_similarity(y,z)), F.log_softmax(z, dim=1), F.log_softmax(y,dim=1), F.log_softmax(y, dim=1)




