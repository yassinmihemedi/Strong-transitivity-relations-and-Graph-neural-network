import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils import test, testc 
def train(model,data,  optimizer,epochs, plot=False):
    train_accuracies, test_accuracies = list(), list()
    for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            out = model(data)
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

            train_acc = test(model,data)
            test_acc = test(model,data, train=False)

            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)
            print('Epoch: {:03d}, Loss: {:.5f}, Train Acc: {:.5f}, Test Acc: {:.5f}'.
                  format(epoch, loss, train_acc, test_acc))

    if plot:
        plt.plot(train_accuracies, label="Train accuracy")
        plt.plot(test_accuracies, label="Validation accuracy")
        plt.xlabel("# Epoch")
        plt.ylabel("Accuracy")
        plt.legend(loc='upper right')
        plt.show()



def trainc(model ,data, Trans_data,optimizer,epochs, plot=False):
    train_accuracies, test_accuracies = list(), list()

    for epoch in range(epochs):
            model.train()

            optimizer.zero_grad()
       
            out, outsim, Trans_out,em1,em2 = model(data, Trans_data)
    

           
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            lossm = F.nll_loss(em2[data.train_mask], data.y[data.train_mask])
      

 
            loss_sim = F.nll_loss(outsim[data.train_mask], Trans_data.y[data.train_mask])
            Trans_loss = F.nll_loss(Trans_out[data.train_mask], Trans_data.y[data.train_mask])

            lossT = loss + Trans_loss #-loss_sim
            lossT.backward()
            
            
            optimizer.step()
       

            train_acc = testc(model, data, Trans_data)
            
            test_acc = testc(model, data, Trans_data, train=False)

            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)
            print('Epoch: {:03d}, Loss: {:.5f}, Train Acc: {:.5f}, Test Acc: {:.5f}'.
                  format(epoch, lossT, train_acc, test_acc))

    if plot:
        plt.plot(train_accuracies, label="Train accuracy")
        plt.plot(test_accuracies, label="Validation accuracy")
        plt.xlabel("# Epoch")
        plt.ylabel("Accuracy")
        plt.legend(loc='upper right')
        plt.show()

    
