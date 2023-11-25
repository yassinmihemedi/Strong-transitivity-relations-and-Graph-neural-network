import argparse
import torch

def Trans_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-3,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=32,
                        help=' hidden dimension .')
    parser.add_argument('--dropout', type=float, default=0.6,
                        help='Dropout rate')
    parser.add_argument('--dataset', type=str, default="cora",
                        help='Dataset to use.')
    parser.add_argument('--model', type=str, default="GCN",
                        choices=[ "GCN","GAT","GATv2", "SGC",],
                        help='model to use.')
    parser.add_argument('--heads', type=int, default=8,
                        choices=[ 2,4,8,16],
                        help='attention heads for GATs models')

    parser.add_argument('--th', type=float, default=0.8,
                        help='threshold of nodes similarity in Trans graph.')
    parser.add_argument('--tr_mask', type=float, default=0.1,
                        help='the portion of nodes for training')
    parser.add_argument('--val_mask', type=float, default=0.2,
                    help='the portion of nodes for validation')
    parser.add_argument('--te_mask', type=float, default=0.5,
                        help='the portion of nodes for test')




    args, _ = parser.parse_known_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args
