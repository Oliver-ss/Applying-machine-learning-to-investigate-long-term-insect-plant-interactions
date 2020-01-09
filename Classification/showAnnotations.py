#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 13:57:39 2019

@author: tomasi
"""

import json, os, datetime
from PIL import Image
from yattag import Doc, indent
from math import ceil

from directories import makeDir, dataRoot, annoRoot, statsDir, imageRoot, fileInfo, readClassTableFromJason


def readDataFromFile(jasonFilename, imageDir, species='', region_attribute='type', expected_file_attributes=''):
    data = []
    with open(jasonFilename, 'r') as file:
        jason = json.load(file)
        imgAnnoList = jason['_via_img_metadata'].values()
        for imgAnno in imgAnnoList:
            imgBasename = imgAnno['filename']
            if len(expected_file_attributes) > 0:
                attribute = imgAnno['file_attributes'][expected_file_attributes]
                if attribute == 'unknown' and expected_file_attributes == 'species':
                    attribute = species
            imgFilename = os.path.join(imageDir, imgBasename)
            image = Image.open(imgFilename)
            regionList = imgAnno['regions']
            for region in regionList:
                rectangle = region['shape_attributes']
                if rectangle['name'] == 'rect':
                    xs, ys = rectangle['x'], rectangle['y']
                    w, h = rectangle['width'], rectangle['height']
                    xe, ye = xs + w, ys + h
                    crop = image.crop((xs, ys, xe, ye))
                    label = region['region_attributes'][region_attribute]
                    record = {'imageFilename': imgFilename,
                              'species': species,
                              'rect': (xs, ys, w, h),
                              'crop': crop, 'label': label}
                    data.append(record)
    return data


def readDataForSpecies(species, numberOfFiles, imageRoot, annoRoot,
                       region_attribute='DamageType', expected_file_attributes=''):
    data = []
    imageDir = os.path.join(imageRoot, species)
        
    for fileNumber in range(1, numberOfFiles + 1):
        annoFilename = os.path.join(annoRoot, species + str(fileNumber) + '.json')
        partial = readDataFromFile(annoFilename, imageDir, species=species, region_attribute=region_attribute,
                                   expected_file_attributes=expected_file_attributes)
        data.extend(partial)
    return data


def mosaic(crops, maxSize=(128, 128), columns = 6):
    n = len(crops)
    rows = ceil(n / columns)
    imgSize = (maxSize[0] * columns, maxSize[1] * rows)
    image = Image.new('RGB', imgSize, (255, 255, 255))
    c, box = 0, [0, 0]
    for crop in crops:
        thumbnail = crop.copy()
        thumbnail.thumbnail(maxSize)
        corner = tuple(box[i] + (maxSize[i] - thumbnail.size[i]) // 2 for i in range(2))
        image.paste(thumbnail, corner)
        c += 1
        if c >= columns:
            c, box = 0, [0, box[1] + maxSize[1]]
        else:
            box = [box[0] + maxSize[0], box[1]]
    return image


def showAnnotations(data, damageTable, htmlDir, timeExt):
        
    doc, tag, text, line = Doc().ttl()
    doc.asis('<!DOCTYPE html>')

    mosaicBasename = '-'.join(('.mosaics', timeExt))
    mosaicDir = os.path.join(htmlDir, mosaicBasename)
    makeDir(mosaicDir)

    speciesList = sorted(set([item['species'] for item in data]))
    labels = sorted(damageTable.keys())
        
    with tag('html'):
        with tag('body'):
            line('h1', 'Annotations on ' + now.strftime("%Y-%m-%d %H:%M"))

            with tag('table', style='width:100%'):
                with tag('tr'):
                    line('th', 'Species')
                    for label in labels:
                        line('th', damageTable[label][1])
                        
                for species in speciesList:
                    with tag('tr'):
                        line('td', species, align='center')
                        for label in labels:
                            count = len([item for item in data \
                                        if item['species'] == species \
                                        and item['label'] == label])
                            line('td', str(count), align='center')
            doc.stag('hr')
            for species in speciesList:
                doc.stag('hr')
                line('h2', species)
                speciesItemList = [item for item in data if item['species'] == species]
                labelList = sorted(set([item['label'] for item in speciesItemList]))

                for label in labelList:
                    line('h3', damageTable[label][1])
                    cropList = [item['crop'] for item in speciesItemList if item['label'] == label]
                    image = mosaic(cropList)
                    imageFilename = '_'.join((species, label, 'mosaic.jpg'))
                    image.save(os.path.join(mosaicDir, imageFilename))
                    srcFilename = os.path.join(mosaicBasename, imageFilename)
                    doc.stag('img', src=srcFilename)
                    doc.stag('hr')

    html = indent(doc.getvalue())
    
    basename = '-'.join(('stats', timeExt)) + '.html'
    filename = os.path.join(htmlDir, basename)
    with open(filename, 'w') as file:
        file.write(html)
    print('Written HTML to file', filename, 'and created mosaic images')


if __name__ == '__main__':
    project = 'equisetum'   # Either 'damage' or 'equisetum'

    if project == 'damage':
        annoFilename = os.path.join(annoRoot, fileInfo[0][0] + '1.json')
        class_type = 'DamageType'
        expected_file_attributes = ''
    elif project == 'equisetum':
        annoRoot = os.path.join(dataRoot, 'Annotations', 'Equisetum')
        annoFilename = os.path.join(annoRoot, 'hyemale1.json')
        class_type = 'type'
        expected_file_attributes = 'species'
        statsDir = os.path.join(annoRoot, 'Stats')
        fileInfo = [('ferrissii', 1), ('hyemale', 1), ('laevigatum', 1)]
    else:
        raise ValueError

    data = []
    for item in fileInfo:
        partial = readDataForSpecies(item[0], item[1], imageRoot, annoRoot,
                                     region_attribute=class_type,
                                     expected_file_attributes=expected_file_attributes)
        data.extend(partial)

    now = datetime.datetime.now()
    timeExt = now.strftime("%Y-%m-%d-%H-%M")
    classTableByString, _ = readClassTableFromJason(annoFilename, class_type=class_type)
    makeDir(statsDir)
    showAnnotations(data, classTableByString, statsDir, timeExt)
