def f1(model, data, dTrans:None):
  from sklearn.metrics import f1_score
  if dTrans==None:

      pred = model(data).max(dim=1)[1]

  else :
    pred,simpred, compred,empred,em2 = model(data, dTrans)
    pred = pred.max(dim=1)[1]

    # print()

  return f1_score(data.y[data.test_mask], pred[data.test_mask],average ='micro'), f1_score(data.y[data.test_mask], pred[data.test_mask],average ='weighted')

def test(model,data, train=True):
    model.eval()

    correct = 0
    pred = model(data).max(dim=1)[1]

    if train:
        correct += pred[data.train_mask].eq(data.y[data.train_mask]).sum().item()
        return correct / (len(data.y[data.train_mask]))
    else:
        correct += pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
        return correct / (len(data.y[data.test_mask]))


def testc(model, data, Trans_data, train=True):
    model.eval()

    correct = 0
    comcorrect = 0
    pred, simpred, compred,empred,em2 = model(data, Trans_data)
    pred = pred.max(dim=1)[1]
    com1 =0


    # em2 = em2.max(dim=1)[1]
    # print('compred',np.argmax(compred,axis=1),compred.size())
    # empred = empred.max(dim =1)[1]


    if train:
        correct += pred[data.train_mask].eq(data.y[data.train_mask]).sum().item()

        correct = correct

        return (correct  / (len(data.y[data.train_mask])))
    else:
        correct += pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()

        return (correct   / (len(data.y[data.test_mask])))