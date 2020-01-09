import torch.optim as optim
from LeavesDataset import dataLoaders
import torch
import torch.nn as nn
from torchvision import models
import pickle
import datetime
from directories import stdRoot, loadClassTableDict
from train import train, test

# Read class names
damageTableByString = loadClassTableDict(stdRoot)['by string']

nClasses = len(damageTableByString)

# Do we have a GPU?
isCuda = torch.cuda.is_available()
device = torch.device("cuda:0" if isCuda else "cpu")
print('Device is', device)

# Adapt resnet18 to our number of classes
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, nClasses)
inputSize = 224
if isCuda:
    model.cuda()
print(model)

# Optimizer
for p in model.parameters():
    p.requires_grad = True
parameters = model.parameters()
# optimizer = optim.Adam(parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
optimizer = optim.SGD(parameters, lr=0.0002, momentum=0.9)

# Loss
criterion = nn.CrossEntropyLoss()

# Do the training
trainedModel, trainHistory, valHistory = train(model, dataLoaders, criterion, optimizer, device, nEpochs=300)

testAccuracy, labels, predictions = test(trainedModel, dataLoaders['testing'], device)

# Save the resulting model
now = datetime.datetime.now()
timeExt = now.strftime("%Y-%m-%d-%H-%M")
filename = 'snapshot-{}.pickle'.format(timeExt)
info = {}
info['model'] = trainedModel
info['training accuracy history'] = trainHistory
info['validation accuracy history'] = valHistory
info['test accuracy'] = testAccuracy
info['labels'] = labels
info['predictions'] = predictions

# This dump records all snapshots over time
with open(filename, 'wb') as f:
    pickle.dump(info, f, pickle.HIGHEST_PROTOCOL)

# This dump makes it easy to find the latest data
with open('snapshot-latest.pickle', 'wb') as f:
    pickle.dump(info, f, pickle.HIGHEST_PROTOCOL)
