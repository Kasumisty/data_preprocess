import os, re

def parseDirs(baseDir, searchPattern):
    """
    :param baseDir: list of base directory
    :param searchPattern: str of re pattern
    :return: list of directories of files in baseDir which satisfies searchPattern

    Examples
    --------
    >>> DIR = ['../data/ace_2005_td_v7/data/Chinese/bn/adj',
        '../data/ace_2005_td_v7/data/Chinese/nw/adj',
        '../data/ace_2005_td_v7/data/Chinese/wl/adj']
    >>> search_pattern = '.apf.xml'
    >>> files_dir = parseDirs(DIR, search_pattern)

    """
    search_pattern = re.compile(searchPattern)
    files_dir = []
    for dir in baseDir:
        file_list = os.listdir(dir)
        files_name = [file_name for file_name in file_list if search_pattern.search(file_name)]
        files_dir += [os.path.join(dir, file_name) for file_name in files_name]
    return files_dir


def loadStopWords(filePath):
    """
    :param filePath: file path of stopwords
    :return: set of stopwords

    """

    with open(filePath, 'r', encoding='utf-8-sig') as f:
        return set([line.strip() for line in f])