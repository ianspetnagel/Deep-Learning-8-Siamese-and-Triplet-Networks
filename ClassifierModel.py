#import torch
#import torch.nn as nn

#class ClassifierModel(nn.Module):

#    def __init__(self):
#        super(ClassifierModel, self).__init__()
#        device = 'cuda' if torch.cuda.is_available() else 'cpu'
#        # load already trained Siamese network
#        self.model_siamese = torch.load('./checkpoint/finalsiamese.tar')
#        # freeze weights and biases of Siamese network
#        for param in self.model_siamese.parameters():
#            param.requires_grad = False
#        self.classifier = nn.Sequential(nn.Linear(2, 50), nn.ReLU(), nn.Linear(50, 10),nn.Softmax(dim=1)).to(device)
#        for param in self.classifier.parameters():
#            param.requires_grad = True
#    def forward(self,x):
#        h = self.model_siamese(x)
#        return self.classifier(h)
