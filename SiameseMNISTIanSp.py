import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from tqdm import tqdm

import MNISTDataSet
from SiameseNetwork import SiameseNetwork
import MNISTDataLoader

def contractive_loss(o1, o2, y):
    g, margin = F.pairwise_distance(o1, o2), 5.0
    loss = (1 - y) * (g ** 2) + y * (torch.clamp(margin - g, min=0) ** 2)
    return torch.mean(loss)

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader, test_loader = MNISTDataSet.get_loaders(args.batch_size)
    model = SiameseNetwork().to(device)
    optimizer = optim.Adam(model.parameters(),lr = 0.001 )
    model.train()
    print("\t".join(["Epoch", "TrainLoss", "TestLoss"]))
    model = model.to(device)
    for e in range(50):
        train_loss, train_n = 0, 0
        for x1, x2, y in tqdm(train_loader, total=len(train_loader)):
            x1, x2 = x1.to(device), x2.to(device)
            y = y.to(device)
            o1, o2 = model(x1), model(x2) # model is invoked twice - Siamese
            loss = contractive_loss(o1, o2, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.data #
            train_n += y.size(0)
        print('epoch=',e,' train loss =',train_loss)
    torch.save(model, "./checkpoint/finalsiamese.tar")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    args = parser.parse_args()
    main(args)