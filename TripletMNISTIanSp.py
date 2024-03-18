import sys
from TripletMNISTDataSet import TripletMNISTDataSet
import TrainTriplet

from torchvision.datasets import MNIST
from torchvision import transforms
from TripletNetwork import EmbeddingNet
from TripletNetwork import TripletNet
from TripletLoss import TripletLoss
import torch
from torch.optim import lr_scheduler
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

cuda = torch.cuda.is_available()
mnist_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728','#9467bd', '#8c564b', '#e377c2', '#7f7f7f','#bcbd22', '#17becf']

def plot_embeddings(embeddings, targets, xlim=None, ylim=None):
    plt.figure(figsize=(10,10))
    for i in range(10):
        inds = np.where(targets==i)[0]
        plt.scatter(embeddings[inds,0], embeddings[inds,1], alpha=0.5,color=colors[i])
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(mnist_classes)
    plt.show()

def extract_embeddings(dataloader, model):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), 2))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:
            if cuda:
                images = images.cuda()
            embeddings[k:k+len(images)] =model.get_embedding(images).data.cpu().numpy()
            labels[k:k+len(images)] = target.numpy()
            k += len(images)
    return embeddings, labels

def main():
    mean, std = 0.1307, 0.3081
    train_dataset = MNIST('./data/MNIST', train=True, download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((mean,), (std,))]))
    test_dataset = MNIST('./data/MNIST', train=False, download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((mean,), (std,))]))

    n_classes = 10
    #--------regular loader for plotting embeddings--------------------
    batch_size = 256
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,shuffle=False, **kwargs)
    #-----------------------------------------------------------------

    triplet_train_dataset = TripletMNISTDataSet(train_dataset)
    triplet_test_dataset = TripletMNISTDataSet(test_dataset)
    batch_size = 128
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    triplet_train_loader = torch.utils.data.DataLoader(triplet_train_dataset,batch_size=batch_size, shuffle=True, **kwargs)
    triplet_test_loader = torch.utils.data.DataLoader(triplet_test_dataset,batch_size=batch_size, shuffle=False, **kwargs)
    # Set up the network and training parameters

    margin = 1.
    embedding_net = EmbeddingNet()
    model = TripletNet(embedding_net)
    if cuda:
        model.cuda()
    loss_fn = TripletLoss(margin)
    lr = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    n_epochs = 50
    log_interval = 100

    TrainTriplet.train(triplet_train_loader, triplet_test_loader, model, loss_fn,optimizer, scheduler, n_epochs, cuda, log_interval)
    train_embeddings_cl, train_labels_cl = extract_embeddings(train_loader, model)
    plot_embeddings(train_embeddings_cl, train_labels_cl)
    val_embeddings_cl, val_labels_cl = extract_embeddings(test_loader, model)
    plot_embeddings(val_embeddings_cl, val_labels_cl)
    

if __name__ == "__main__":
    sys.exit(int(main() or 0))