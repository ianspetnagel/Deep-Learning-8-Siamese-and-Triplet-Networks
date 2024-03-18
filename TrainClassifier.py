#import sys
#import torch
#import torch.nn as nn
#import MNISTDataLoader
#import torch.optim as optim
#import torch.nn.functional as F

#from ClassifierModel import ClassifierModel

#def main():
#    device = 'cuda' if torch.cuda.is_available() else 'cpu'
#    model = ClassifierModel()
#    # -----------------------test the accuracy---------------
#    def test(model, test_loader):
#        model.eval()
#        test_loss = 0
#        correct = 0
#        lossc = nn.CrossEntropyLoss()
#        with torch.no_grad():
#            for data, target in test_loader:
#                data = data.to(device)
#                target = target.to(device)
#                output = model(data)
#                test_loss = lossc(output, target.to(device))
#                pred = output.data.max(1, keepdim=True)[1]
#                correct += pred.eq(target.data.view_as(pred)).sum()
#        test_loss /= len(test_loader.dataset)
#        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))

#    # ---------train the added classifier----------------
#    log_interval = 10
#    n_epochs = 10
#    opt = optim.Adam(model.parameters(),lr = 0.0002 )
#    train_loader, test_loader = MNISTDataLoader.get_loader_normal()
#    lossc = nn.CrossEntropyLoss()
#    for epoch in range(1, n_epochs):
#        for batch_idx, (data, target) in enumerate(train_loader):
#            opt.zero_grad()
#            output = model(data.to(device))
#            loss = lossc(output, target.to(device))
#            loss.backward()
#            opt.step()
#            if batch_idx % log_interval == 0:
#                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),100. * batch_idx / len(train_loader), loss.item()))
    
                
#if __name__ == "__main__":
#    sys.exit(int(main() or 0))