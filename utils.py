import os, re


def parseDirs(baseDir, searchPattern):
    search_pattern = re.compile(searchPattern)
    files_dir = []
    for dir in baseDir:
        file_list = os.listdir(dir)
        files_name = [file_name for file_name in file_list if search_pattern.search(file_name)]
        files_dir += [os.path.join(dir, file_name) for file_name in files_name]
    return files_dir


def loadStopWords(filePath):
    with open(filePath, 'r', encoding='utf-8-sig') as f:
        return set([line.strip() for line in f])
