import pickle
import torch
from LeavesDataset import dataLoaders
from train import test
from directories import listDir

pickleFilenames = listDir('.', '.pickle')

for filename in pickleFilenames:
    print('Patching', filename)

    with open(filename, 'rb') as f:
        info = pickle.load(f)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    accuracy, labels, predictions = test(info['model'], dataLoaders['testing'], device)

    info['accuracy'] = accuracy
    info['labels'] = labels
    info['predictions'] = predictions

    patchFilename = 'patched-' + filename
    print('Patch in', patchFilename)
    with open(patchFilename, 'wb') as f:
        pickle.dump(info, f, pickle.HIGHEST_PROTOCOL)
