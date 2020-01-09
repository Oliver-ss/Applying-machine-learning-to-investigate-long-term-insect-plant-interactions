import json, datetime
import os, shutil
from skimage import io
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
import numpy as np
from math import floor
import warnings
import pickle
from directories import annoRoot, imageRoot, stdRoot, fileInfo, makeDir, nameComponents, deepenDir, \
    readDamageTableFromJason, saveDamageTableDict

warnings.filterwarnings('ignore')


def readDataFromFile(jasonFilename, imageDir, species=''):
    data = []
    with open(jasonFilename, 'r') as file:
        jason = json.load(file)
        imgAnnoList = jason['_via_img_metadata'].values()
        for imgAnno in imgAnnoList:
            imgBasename = imgAnno['filename']
            imgFilename = os.path.join(imageDir, imgBasename)
            regionList = imgAnno['regions']
            for region in regionList:
                rectangle = region['shape_attributes']
                if rectangle['name'] == 'rect':
                    xs, ys = rectangle['x'], rectangle['y']
                    w, h = rectangle['width'], rectangle['height']
                    xe, ye = xs + w, ys + h
                    label = region['region_attributes']['DamageType']
                    record = {'imageFilename': imgFilename,
                              'species': species, 'label': label,
                              'rect': (xs, ys, w, h)}
                    data.append(record)
    return data


def readDataByImage(fileInfo, imageRoot, annoRoot):
    def readDataForSpecies(species, numberOfFiles, imageRoot, annoRoot):
        data = []
        imageDir = os.path.join(imageRoot, species)

        for fileNumber in range(1, numberOfFiles + 1):
            annoFilename = os.path.join(annoRoot, species + str(fileNumber) + '.json')
            partial = readDataFromFile(annoFilename, imageDir, species=species)
            data.extend(partial)
        return data

    def byImage(data):
        dict = {}
        for record in data:
            info = record.copy()
            filename = info.pop('imageFilename')
            try:
                dict[filename].append(info)
            except KeyError:
                dict[filename] = [info]
        return dict

    data = []
    for item in fileInfo:
        partial = readDataForSpecies(item[0], item[1], imageRoot, annoRoot)
        data.extend(partial)

    dataDict = byImage(data)
    return dataDict


# def extractData(data, damageTable, dataDir, timeExt, thumbBasename):
#     thumbDir = os.path.join(dataDir, thumbBasename)
#     makeDir(thumbDir)
#     sizeFilename = os.path.join(thumbDir, 'sizes.txt')
#     with open(sizeFilename, 'w') as sizeFile:
#
#         speciesList = sorted(set([item['species'] for item in data]))
#         labels = sorted(damageTable.keys())
#
#         for species in speciesList:
#             speciesItemList = [item for item in data if item['species'] == species]
#             labelList = sorted(set([item['label'] for item in speciesItemList]))
#             speciesDir = os.path.join(thumbDir, species)
#             makeDir(speciesDir)
#
#             for label in labelList:
#                 thumbList = [item['thumbnail'] for item in speciesItemList if item['label'] == label]
#                 labelDir = os.path.join(speciesDir, label)
#                 makeDir(labelDir)
#                 thumbDict = {}
#                 for item in speciesItemList:
#                     if item['label'] == label:
#                         path, name = os.path.split(item['imageFilename'])
#                         base, ext = os.path.splitext(name)
#                         try:
#                             thumbNum = thumbDict[base]
#                         except KeyError:
#                             thumbNum = 0
#                         thumbFilename = '{}_{}{}'.format(os.path.join(labelDir, base), thumbNum, ext)
#                         item['thumbnail'].save(thumbFilename)
#                         sizeFile.write('{} {} {}\n'.format(item['rect'][2], item['rect'][3], label))
#                         thumbDict[base] = thumbNum + 1


def sampleSwatches(thumbnail, image, rect, swatchSide, stride, satThreshold, maxBackgroundFraction, label):

    def littleBackground(swatch, thr, frac, label):
        if label == 'margin' or label == 'normmar' or label == 'interior':
            sat = rgb2hsv(swatch)[:, :, 1]
            lowSatCount = (sat < thr).sum()
            return lowSatCount < frac * swatch.size
        else:
            return True

    if swatchSide is None:
        # Only swatch is thumbnail itself
        return [thumbnail]
    else:
        cs, rs, w, h = rect
        cf, rf = cs + w, rs + h
        crop = [[rs, rf], [cs, cf]]

        # Make the thumbnail big enough to contain a swatch, if needed
        for dim in range(2):
            thumbnailSide = thumbnail.shape[dim]
            defect = max(0, swatchSide - thumbnailSide)
            before = floor(defect/2)
            after = defect - before
            if defect:
                crop[dim][0] -= before
                crop[dim][1] += after

                # Recede back into the image, if needed
                if crop[dim][0] < 0:
                    excess = -crop[dim][0]
                    for i in range(2):
                        crop[dim][i] += excess
                if crop[dim][1] > image.shape[dim]:
                    excess = crop[dim][1] - image.shape[dim]
                    for i in range(2):
                        crop[dim][i] -= excess
        rs, rf, cs, cf = crop[0][0], crop[0][1], crop[1][0], crop[1][1]
        thumbnail = image[rs:rf, cs:cf, :]

        source = []
        for dim in range(2):
            thumbnailSide = thumbnail.shape[dim]
            source.append(list(range(0, thumbnailSide - swatchSide + 1, stride)))

        swatches = []
        for srs in source[0]:
            srf = srs + swatchSide
            for scs in source[1]:
                scf = scs + swatchSide
                swatch = thumbnail[srs:srf, scs:scf, :]
                if littleBackground(swatch, satThreshold, maxBackgroundFraction, label):
                    swatches.append(swatch)

        return swatches


def makeThumbnailsAndSwatches(data, fileInfo, damageTableByString, damageTableByNumber, dataRoot, swatchSide=112, stride=15,
             maxBackgroundFraction=0.6, debug=True):

    def cropDirs(dir, record):
        thumbDir = os.path.join(dir, record['species'], record['label'])
        swatchDir = os.path.join(thumbDir, 'swatches')
        return thumbDir, swatchDir

    def carve(image, rectangle):
        xs, ys = rectangle[0], rectangle[1]
        xf, yf = xs + rectangle[2], ys + rectangle[3]
        return image[ys:yf, xs:xf, :]

    def newIndex(key, dct):
        try:
            idx = dct[key] + 1
        except KeyError:
            idx = 0
        dct[key] = idx
        return idx

    speciesList = [item[0] for item in fileInfo]
    labels = damageTableByString.keys()

    # Set up directory tree

    print('Clearing output directory', dataRoot)
    if os.access(dataRoot, os.F_OK):
        shutil.rmtree(dataRoot)
    makeDir(dataRoot)
    print('Done. Processing annotations:')

    if debug:
        backDir = deepenDir(dataRoot, 'backgrounds')

    saveDamageTableDict(dataRoot, damageTableByString, damageTableByNumber)

    dataDir = deepenDir(dataRoot, 'samples')

    for species in speciesList:
        speciesDir = deepenDir(dataDir, species)
        if debug:
            deepenDir(backDir, species)
        for label in labels:
            labelDir = deepenDir(speciesDir, label)
            deepenDir(labelDir, 'swatches')

    thumbCount, swatchCount = 0, 0
    imageFilenameList = data.keys()
    nImages = len(imageFilenameList)
    imageCount = 0
    for imageFilename in imageFilenameList:
        imageCount += 1
        print('{:6d}'.format(imageCount), '/', nImages, ': ', imageFilename, sep='')
        image = io.imread(imageFilename)
        _, imageBasename, ext = nameComponents(imageFilename)

        # Compute a saturation threshold for this image to filter out swatches with too much background
        saturation = rgb2hsv(image)[:, :, 1]
        satThresh = threshold_otsu(saturation)

        if debug:
            # Save the background image
            background = ((saturation < satThresh) * 255).astype(np.uint8)
            _, imageBasename, imageExt = nameComponents(imageFilename)
            species = data[imageFilename][0]['species']
            satPath = os.path.join(backDir, species, imageBasename + '-sat' + imageExt)
            backPath = os.path.join(backDir, species, imageBasename + '-bgd' + imageExt)
            io.imsave(satPath, saturation)
            io.imsave(backPath, background)

        thumbIndices = {}
        for record in data[imageFilename]:
            rect, label = record['rect'], record['label']
            thumbDir, swatchDir = cropDirs(dataDir, record)
            thumbIndex = newIndex(thumbDir, thumbIndices)
            thumbFilename = os.path.join(thumbDir, '{}_{}{}'.format(imageBasename, thumbIndex, ext))
            thumbnail = carve(image, rect)
            io.imsave(thumbFilename, thumbnail)
            thumbCount += 1

            swatches = sampleSwatches(thumbnail, image, rect, swatchSide, stride, satThresh,
                                      maxBackgroundFraction, label)
            swatchIndices = {}
            for swatch in swatches:
                swatchIndex = newIndex(swatchDir, swatchIndices)
                swatchFilename = os.path.join(swatchDir, '{}_{}_{}{}'.format(imageBasename, thumbIndex, swatchIndex, ext))
                io.imsave(swatchFilename, swatch)
                swatchCount += 1

    print('Made', thumbCount, 'thumbnails and', swatchCount, 'swatches in', dataDir)
    if debug:
        print('Saved saturation and background images in', backDir)


annoFilename = os.path.join(annoRoot, fileInfo[0][0] + '1.json')
damageTableByString, damageTableByNumber = readDamageTableFromJason(annoFilename)

data = readDataByImage(fileInfo, imageRoot, annoRoot)
makeThumbnailsAndSwatches(data, fileInfo, damageTableByString, damageTableByNumber, stdRoot, swatchSide=224, stride=20)
