import os
import numpy as np
from torchvision import transforms
import warnings
from directories import stdRoot, annoRoot, listDir, pathItems, labelFromPath, loadClassTableDict
from PIL import Image
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
from torch import tensor

warnings.filterwarnings('ignore')

stdTransform = transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


class LeavesDataset(Dataset):
    """Leaves data set"""

    def __init__(self, rootDir, annoRoot, imageExtension='.jpeg', transform=stdTransform):
        self.rootDir = rootDir
        self.imageExtension = imageExtension
        self.transform = transform
        self.items = []

        tables = loadClassTableDict(rootDir)
        self.damageTableByString = tables['by string']

        samplesDir = os.path.join(rootDir, 'samples')
        speciesList = listDir(samplesDir)
        for species in speciesList:
            speciesDir = os.path.join(samplesDir, species)
            labels = listDir(speciesDir)
            for label in labels:
                swatchDir = os.path.join(speciesDir, label, 'swatches')
                swatchNames = listDir(swatchDir, self.imageExtension)
                for swatchName in swatchNames:
                    swatchFile = os.path.join(swatchDir, swatchName)
                    self.items.append(swatchFile)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        swatchFile = self.items[index]
        swatch = Image.open(swatchFile)
        if self.transform:
            swatch = self.transform(swatch)
        label = tensor(labelFromPath(swatchFile, self.damageTableByString))
        return swatch, label


def split(data, level, fractions, batchSize):
    def groupId(filename, level):
        comp = pathItems(filename)
        if level == 'species':
            id = comp['species']
        elif level == 'image':
            id = (comp['species'], comp['image'])
        elif level == 'thumbnail':
            id = (comp['species'], comp['image'], comp['thumbnail'])
        elif level == 'swatch':
            id = (comp['species'], comp['image'], comp['thumbnail'], comp['swatch'])
        else:
            raise ValueError('invalid split level')
        return id

    # To do: we currently partition the number of items at the given level uniformly. For instance, the numbers of
    # THUMBNAILS in the partition are proportional to the given fractions. The proper thing is to partition the
    # number of items so that the numbers of SWATCHES are proportional to the given fractions. If thumbnail sizes
    # are uniformly distributed, the difference is not huge.
    def partition(nGroups, fractions):
        if len(fractions) == 2:
            fractions.append(1 - sum(fractions))

        fractions = np.array(fractions)
        if len(fractions) != 3 or np.abs(np.sum(fractions) - 1) > 1e-6 or np.any(fractions < 0):
            msg = 'fractions must be a list of three non-negative numbers that add up to 1'
            msg += '\nor of two non-negative numbers that add up to at most 1'
            raise ValueError(msg)

        nIndices = np.round(fractions * nGroups)
        while np.sum(nIndices) > nGroups:
            nIndices[np.argmax(nIndices)] -= 1
        while np.sum(nIndices) < nGroups:
            nIndices[np.argmin(nIndices)] += 1

        for k in range(len(nIndices)):
            if fractions[k] > 0 and nIndices[k] <= 0:
                msg = '{} groups are not enough to produce the desired split {}'
                raise ValueError(msg.format(nGroups, fractions))

        return nIndices.astype(int).tolist()

    group = {}
    for index in range(len(data)):
        filename = data.items[index]
        key = groupId(filename, level)
        try:
            group[key].append(index)
        except KeyError:
            group[key] = [index]

    groupItems = list(group.items())
    nGroups = len(groupItems)
    randomSeed = 37
    np.random.seed(randomSeed)
    np.random.shuffle(groupItems)

    nTrain, nVal, nTest = partition(nGroups, fractions)
    thresholds = [nTrain, nTrain + nVal]
    subGroupItems = [groupItems[:thresholds[0]], groupItems[thresholds[0]:thresholds[1]], groupItems[thresholds[1]:]]

    loaders = []
    for sub in range(3):
        swatchIndices = []
        for _, groupValue in subGroupItems[sub]:
            swatchIndices.extend(groupValue)
        sampler = SubsetRandomSampler(swatchIndices)
        loaders.append(DataLoader(data, batch_size=batchSize, sampler=sampler))

    return {'training': loaders[0], 'validation': loaders[1], 'testing': loaders[2]}


data = LeavesDataset(stdRoot, annoRoot)


# Training, validation, and testing data loaders, returned as a dictionary of three loaders
batchSize = 50
level = 'thumbnail'
dataFractions = [0.6, 0.2]
dataLoaders = split(data, level, dataFractions, batchSize)

length = [len(loader.sampler.indices) for loader in dataLoaders.values()]
msg = 'Data split with {} training samples, {} validation samples, and {} test samples'
print(msg.format(length[0], length[1], length[2]))
