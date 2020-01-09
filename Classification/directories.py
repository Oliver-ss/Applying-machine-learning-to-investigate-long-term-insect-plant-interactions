import os, errno, json, pickle

userDir = os.path.expanduser('~')
projectRoot = os.path.join(userDir, 'Dropbox', 'Leaves')
dataRoot = os.path.join(projectRoot, 'Data')
annoRoot = os.path.join(dataRoot, 'Annotations', 'Damage')
statsDir = os.path.join(annoRoot, 'Stats')
imageRoot = os.path.join(dataRoot, 'Images')
stdRoot = os.path.join(dataRoot, 'Standardized')


def makeDir(name):
    try:
        os.mkdir(name)
    except OSError as e:
        if e.errno != errno.EEXIST:
            print('Cannot create directory', name)
            raise


def listDir(dir, extension=None):
    if os.access(dir, os.F_OK):
        allFiles = os.listdir(dir)
        if extension is None:
            files = [file for file in allFiles if file[0] != '.']
        else:
            files = []
            for file in allFiles:
                _, ext = os.path.splitext(file)
                if ext == extension and file[0] != '.':
                    files.append(file)
        return files

    else:
        print('Cannot access directory', dir)
        raise ValueError


def nameComponents(name):
    path, base = os.path.split(name)
    base, ext = os.path.splitext(base)
    return path, base, ext


def deepenDir(root, leaf):
    dir = os.path.join(root, leaf)
    makeDir(dir)
    return dir


def readClassTableFromJason(jasonFilename, class_type='type'):
    with open(jasonFilename, 'r') as file:
        jason = json.load(file)
        classTableByString = jason['_via_attributes']['region'][class_type]['options']
        classNumber = 0
        classTableByNumber = {}
        for key in classTableByString.keys():
            classTableByString[key] = (classNumber, classTableByString[key])
            classTableByNumber[classNumber] = key
            classNumber += 1

        return classTableByString, classTableByNumber


labelTableFileName = 'label_tables.pickle'


def saveClassTableDict(dataRoot, classTableByString, classTableByNumber):
    tablesPickleFilename = os.path.join(dataRoot, labelTableFileName)
    labelTables = {'by string': classTableByString, 'by number': classTableByNumber}
    with open(tablesPickleFilename, 'wb') as file:
        pickle.dump(labelTables, file, pickle.HIGHEST_PROTOCOL)


def loadClassTableDict(dataRoot):
    tableFilename = os.path.join(dataRoot, labelTableFileName)
    with open(tableFilename, 'rb') as file:
        classTables = pickle.load(file)
    return classTables  # dict keys are 'by string' and 'by number'


def pathItems(name):
    separator = os.sep
    components = name.split(separator)
    swatchPos = components.index('swatches')
    labelPos, speciesPos = swatchPos - 1, swatchPos - 2
    items = {}
    items['root'] = os.path.join(*components[:speciesPos])
    items['species'], items['label'] = components[speciesPos], components[labelPos]
    fileName = components[-1]
    base, extension = os.path.splitext(fileName)
    items['image'], items['thumbnail'], items['swatch'] = base.split('_')
    return items


def labelFromPath(path, table):
    label = pathItems(path)['label']
    return table[label][0]


fileInfo = [('OnocleaSensibilis', 3), ('QuercusBicolor', 4)]
