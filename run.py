import torch
from Trans_gnn import Trans_GCN, Trans_GAT,Trans_GATV2,Trans_SGC
from Base_gnn import GCN,GAT,GATV2,SGC
from data_loader import ddataset,data_mask,cluster,Trans_graph
from train import train,trainc
from utils import f1
from args import Trans_args
if __name__ == "__main__":

   args = Trans_args()
   print(args)
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
  #  dname= ['USA','cora', 'citeseer', 'pubmed', 'actor', 'twitch']
   dname= [args.dataset]

   for dn in dname:
    dataset = ddataset(dn)
    data = dataset[0].to(device)
    # data = data_mask(data, 0.1, 0.2, 0.5)

    # data.edge_index, edge_mask = dropout_edge(dataedge, p=p)

    if dn not in ['cora', 'citeseer', 'pubmed']:
        data = data_mask(data,args.tr_mask,args.val_mask, args.te_mask).to(device)
        H1 = Trans_graph(data, args.th,1)
        dTrans = cluster(H1,dataset).to(device)

    else :
        H1 = Trans_graph(data, args.th,1)
        dTrans = cluster(H1,dataset).to(device)

       




    if args.model =="GCN":
       modelc = Trans_GCN(dataset,args.hidden).to(device)
       model = GCN(dataset,args.hidden).to(device)
    
    elif args.model =="GAT":
       modelc = Trans_GAT(dataset,args.hidden, args.heads).to(device)
       model = GAT(dataset,args.hidden,args.heads).to(device)

    elif args.model =="GATv2":
       modelc = Trans_GATV2(dataset,args.hidden, args.heads).to(device)
       model = GATV2(dataset,args.hidden, args.heads).to(device)

    elif args.model == "SGC":
       modelc = Trans_SGC(dataset,args.hidden).to(device)
       model = SGC(dataset,args.hidden).to(device)
       
    
       
       

    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr, weight_decay=args.weight_decay)



    optimizerc = torch.optim.Adam(modelc.parameters(),lr=args.lr, weight_decay=args.weight_decay)
    print('Trans gnn model start')
    trainc(modelc, data,dTrans ,optimizerc,epochs=args.epochs, plot=False)
    print('Trans gnn model end\n')
    
    print('base gnn model start')
    train(model, data,optimizer,epochs=args.epochs, plot=False)
    print('base model gnn end\n')
    
    fscorec = f1(modelc, data, dTrans=dTrans)
    print('Trans gnn model f_1 score  (micro, weighted)', fscorec)
    fscore = f1(model, data, dTrans=None)
    print('Base gnn model f_1 score  (micro, weighted)', fscore)


