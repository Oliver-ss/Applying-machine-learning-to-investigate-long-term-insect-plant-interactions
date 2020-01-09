import sys
import time
import torch
import copy

def train(model, dataLoaders, criterion, optimizer, device, nEpochs=25):
    since = time.time()
    trainHistory, valHistory = [], []
    bestWeights = copy.deepcopy(model.state_dict())
    bestAccuracy = 0.0
    bestEpoch = 0

    print('Training')

    for epoch in range(nEpochs):
        print('Epoch {}/{}'.format(epoch, nEpochs - 1))
        print('-' * 10)

        for phase in ['training', 'validation']:
            if phase == 'training':
                model.train()
            else:
                model.eval()

            runningLoss, runningCorrects = 0.0, 0

            batch = 0
            print('{} batch'.format(phase), end=' ')
            for inputs, labels in dataLoaders[phase]:
                if batch % 10 == 0:
                    print('{}/{}'.format(batch, len(dataLoaders[phase])), end=' ')
                    sys.stdout.flush()
                batch += 1
                if batch % 200 == 0:
                    print()

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'training'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, predictions = torch.max(outputs, 1)

                    # Backward pass if needed
                    if phase == 'training':
                        loss.backward()
                        optimizer.step()

                runningLoss += loss.item() * inputs.size(0)
                runningCorrects += torch.sum(predictions == labels.data)

            nSamples = len(dataLoaders[phase].sampler.indices)
            epochLoss = runningLoss / nSamples
            epochAccuracy = float(runningCorrects) / nSamples

            print('\n{} loss: {:.4f}; accuracy: {:.4f}'.format(phase, epochLoss, epochAccuracy), end='')

            if phase == 'training':
                trainHistory.append(epochAccuracy)
                print()
            elif phase == 'validation':
                if epochAccuracy > bestAccuracy:
                    bestAccuracy = epochAccuracy
                    bestEpoch = epoch
                    bestWeights = copy.deepcopy(model.state_dict())
                valHistory.append(epochAccuracy)
                print(' (best validation accuracy so far is {:.4f} after epoch {})'.format(bestAccuracy, bestEpoch))

        print()

    timeElapsed = time.time() - since
    print('Training time {:.0f}m {:.0f}s'.format(timeElapsed // 60, timeElapsed % 60))
    print('Best validation accuracy: {:4f}'.format(bestAccuracy))

    model.load_state_dict(bestWeights)
    return model, trainHistory, valHistory


def test(model, testLoader, device):
    print('Testing')
    model.eval()
    corrects = 0

    allLabels, allPredictions = [], []
    batch = 0
    for inputs, labels in testLoader:
        if batch % 10 == 0:
            print('{}/{}'.format(batch, len(testLoader)), end=' ')
            sys.stdout.flush()
        batch += 1
        if batch % 200 == 0:
            print()

        inputs = inputs.to(device)
        labels = labels.to(device)
        allLabels.extend(labels.tolist())

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)
            allPredictions.extend(predictions.tolist())

        corrects += torch.sum(predictions == labels.data)

    nSamples = len(testLoader.sampler.indices)
    accuracy = float(corrects) / nSamples

    print('\nTest accuracy on {} samples: {:.4f}'.format(nSamples, accuracy))

    return accuracy, allLabels, allPredictions
