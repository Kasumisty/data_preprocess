# try to find whether each sentebnce has a trigger
from collections import defaultdict
file = '../processed_data/predata.txt'
with open(file, 'r', encoding='utf-8') as f:
    data = f.readlines()

numofTrigger = defaultdict(int)
count = 0
all = 0

l = []

for line in data:
    l.append(line.strip())
    if line == '\n':
        # print('true')
        all += 1
        numofTrigger[count] += 1
        if count == 0:
            print(l)
        l = []
        count = 0
    if len(line.split('\t')) > 1:
        count += 1

a = 3
print('num of sentences which dont have trigger:', numofTrigger[0])
print('num of sentences which have {0} trigger(s):'.format(a), numofTrigger[a])
print('num of all sentences:', all)