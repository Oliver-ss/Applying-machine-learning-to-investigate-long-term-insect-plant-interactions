import pickle
import torch
import os
from directories import dataRoot, fileInfo, loadClassTableDict
import numpy as np
from yattag import Doc, indent


with open('patched-snapshot-latest.pickle', 'rb') as f:
    info = pickle.load(f)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

accuracy, labels, predictions = info['accuracy'], info['labels'], info['predictions']

# Read class names
tables = loadClassTableDict(dataRoot)
damageTableByString, damageTableByNumber = tables['by string'], tables['by number']
nClasses = len(damageTableByString)

confusion = np.zeros((nClasses, nClasses), dtype=np.int32)
for label, prediction in zip(labels, predictions):
    confusion[label, prediction] += 1

numbers = list(range(nClasses))

stripUndefined = True
if stripUndefined:
    numbers = numbers[1:]

names = [damageTableByNumber[number] for number in numbers]

doc, tag, text, line = Doc().ttl()
doc.asis('<!DOCTYPE html>')
with tag('html'):
    with tag('head'):
        with tag('style'):
            text('table, th, td {border: 1px solid gray; border-collapse: collapse;} th, td {padding: 5px;}')

    with tag('body'):
        with tag('table', style='width:100%'):
            with tag('tr'):
                line('th colspan=2', '')
                line('th colspan="{}"'.format(len(numbers)), 'Predicted')

            with tag('tr'):
                line('th colspan=2', '')
                for name in names:
                    line('th', name)

            for trueNumber in numbers:
                with tag('tr'):
                    if trueNumber == numbers[0]:
                        line('th rowspan={}'.format(len(numbers)), 'True')
                    line('th', damageTableByNumber[trueNumber], align='right')
                    for predictedNumber in numbers:
                        line('td', str(confusion[trueNumber, predictedNumber]), align='center')

        doc.stag('p')
        line('p', 'Overall accuracy {:.2f} percent'.format(accuracy * 100))

html = doc.getvalue() #indent(doc.getvalue())
filename = 'confusion.html'
with open(filename, 'w') as file:
    file.write(html)
print('Written HTML to file', filename)
