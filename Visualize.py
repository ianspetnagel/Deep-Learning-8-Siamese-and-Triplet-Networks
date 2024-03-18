#import argparse
#import sys
#import torch
#from torch.autograd import Variable
#from torchvision import datasets
#from torchvision import transforms
#import numpy as np
#from tqdm import tqdm
#import matplotlib
#matplotlib.use("Agg")
#import matplotlib.pyplot as plt
#from matplotlib import offsetbox
#import MNISTDataLoader
#from SiameseNetwork import SiameseNetwork
#import matplotlib.pyplot as plt


#def main():
#    device = 'cuda' if torch.cuda.is_available() else 'cpu'
#    model = torch.load('./checkpoint/finalsiamese.tar')
#    train_loader, test_loader = MNISTDataLoader.get_loader_normal()

#    model.eval()
#    inputs, embs, targets = [], [], []
#    for x, t in tqdm(test_loader, total=len(test_loader)):
#        x = x.to('cuda')
#        t = t.to('cuda')
#        o1 = model(x)
#        embs.append(o1.cpu().data.numpy())
#        targets.append(t.cpu().numpy())
#    embed = np.array(embs).reshape((-1, 2)) # outside of for loop
#    targets = np.array(targets).reshape((-1,))

#    labelset = set(targets.tolist())
#    fig = plt.figure(figsize=(8,8))
#    ax = fig.add_subplot(111)
#    for label in labelset:
#        indices = np.where(targets == label)
#        ax.scatter(embed[indices,0], embed[indices,1], label = label, s = 20)
#        ax.legend()
#        fig.savefig('embed.jpeg', format='jpeg', dpi=600, bbox_inches='tight')
#    plt.plot()
#    plt.draw()

#if __name__ == "__main__":
#    sys.exit(int(main() or 0))
