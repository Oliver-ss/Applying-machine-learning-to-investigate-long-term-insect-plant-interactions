#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 12:57:33 2019

@author: tomasi
"""

import csv, requests, sys, argparse, os, errno

def download(csvFilePath, downloadBasePath,
             skipRows=1, nameCol=0, urlCol=1):
    def makeDir(name):
        try:
            os.mkdir(name)
        except OSError as e:
            if e.errno != errno.EEXIST:
                print('Cannot create directory', name)
                raise
                
    (downloadRoot, _) = os.path.splitext(os.path.basename(csvFilePath))
    downloadPath = downloadBasePath + os.sep + downloadRoot
                
    if not os.path.exists(downloadBasePath):
        makeDir(downloadBasePath)
        
    if not os.path.exists(downloadPath):
        makeDir(downloadPath)
        
    print('Downloading images from', csvFilePath,
          'to directory', downloadPath)
    print('Reading image names from column', nameCol,
          'and URLs from column', urlCol)

    with open('skipLog.txt', 'w') as skipLog:
        with open(csvFilePath) as file:
            rows = csv.reader(file, delimiter=',', quotechar='"')

            # Skip header row(s)
            for k in range(skipRows):
                next(rows, None)

            for row in rows:
                imgBasename = str(row[nameCol])
                imgFilename = downloadPath + os.sep + imgBasename
                url = row[urlCol]
                try:
                    result = requests.get(url, stream=True, timeout=5)
                    if result.status_code:
                        try:
                            ctype = result.headers['Content-Type']
                        except KeyError:
                            ctype = ''
                        suffix = ctype.split('/')
                        if suffix[0] == 'image':
                            suffix = '.' + suffix[1] if len(suffix) > 1 else ''
                            imgFilename += suffix
                            try:
                                image = result.raw.read()
                                open(imgFilename, "wb").write(image)
                                urlPrefix = url.split('?')[0]
                                print(urlPrefix, '==>', imgFilename)
                            except:
                                print('[Download failed for CoreId', imgBasename, ']')
                        else:
                            print('[Skipping CoreId', imgBasename, 'which links to',
                                    suffix[0], 'rather than image]')
                    else:
                        print('[Download of', imgBasename, 'failed]')
                except requests.Timeout:
                    skipLog.write(imgBasename + '\n')
                    skipLog.flush()
                    print('Skipping CoreId', imgBasename, 'because of timeout')

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--namecol", type=int, default=0,
        help="CSV column number (starting at 0) with the file names (default 0)")
    parser.add_argument("-u", "--urlcol", type=int, default=1,
        help="CSV column number (starting at 0) with the URLs (default 1)")
    parser.add_argument("-s", "--skiprows", type=int, default=1,
        help="Number of header rows to skip (default 1)")
    parser.add_argument("filename", help="name of the CSV file to read")
    parser.add_argument("directory", default='Downloads',
        help="top-level directory with all the images")
    args = parser.parse_args()
    nameCol = args.namecol
    urlCol = args.urlcol
    skipRows = args.skiprows
    filename = args.filename
    directory = args.directory
    
    download(filename, directory, skipRows=skipRows, nameCol=nameCol,
             urlCol=urlCol)

if __name__ == "__main__":
    main(sys.argv)
#else:
#    download('../Data/images2.csv', '../Data/Downloads')
