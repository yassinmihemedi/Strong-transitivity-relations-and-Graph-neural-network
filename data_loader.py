import torch
from torch_geometric.datasets import Planetoid, Twitch,Actor,Airports
from torch_geometric.loader import ClusterData, ClusterLoader
from torch_geometric.utils import to_networkx, from_networkx
import networkx as nx


def ddataset(name):

  if name=='USA':
    dataset = Airports(root=r'./Airports', name='USA')

  if name=='cora':
    print('cora')
    dataset = Planetoid(root=r'./cora', name='cora')

  if name=='citeseer':
    print('citeseer')
    dataset = Planetoid(root=r'./citeseer', name='citeseer')

  if name=='pubmed':
    print('pubmed')
    dataset = Planetoid(root=r'./pubmed', name='pubmed')

  if name=='actor':
    dataset = Actor(root=r'./actor')

  if name=='twitch':
    dataset = Twitch(root=r'./Twitch', name='PT')
  return dataset


def data_mask(data,  tr_sp, v_sp,te_sp):
  data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool )

  data.train_mask[:int(tr_sp*data.num_nodes)] = True


  # print('2',len(np.where(data.train_mask == True)[0]) )
  data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool )

  data.val_mask[int(tr_sp*data.num_nodes) : int(v_sp*data.num_nodes) ] = True


  data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool )

  data.test_mask[ int(te_sp*data.num_nodes):  ] = True
  data = data
  return data

def Trans_graph(data, threshold, T ):

  print()
  print('starting transitive graph construction')
  print()
  if T==1:
    Gsim = to_networkx(data, to_undirected=True)

    simrank = nx.simrank_similarity(Gsim)

    threshold = threshold

    H = nx.Graph()
    H.add_nodes_from(Gsim.nodes())
    for u in H.nodes():
      for v in H.nodes():
        if  u!=v and simrank[u][v]> threshold:
                H.add_edge(u, v)
    print('#edges',H.number_of_edges())
    print()
    print('Transitive graph construction end')
    print()
    H1 = from_networkx(H)
    H1.x = data.x.clone()
    H1.y = data.y.clone()
    H1.train_mask = data.train_mask.clone()
    H1.test_mask = data.test_mask.clone()



  return H1

def cluster(data,dataset):
    cluster_dataTrans = ClusterData(data, num_parts=dataset.num_classes+1, recursive=False,
                            )
    train_loaderTrans = ClusterLoader(cluster_dataTrans, batch_size=dataset.num_classes+1, shuffle=True,
                                num_workers=2)
    for d_Trans in train_loaderTrans:
        d_Trans = d_Trans

    return d_Trans