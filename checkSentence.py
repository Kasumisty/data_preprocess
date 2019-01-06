# try to find whether each sentebnce has a trigger
file = '../processed_data/predata.txt'
with open(file, 'r', encoding='utf-8') as f:
    data = f.readlines()

tag = 1
count = 0
all = 0
for line in data:
    if line == '\n':
        # print('true')
        all += 1
        if tag:
            print('err')
            count += 1
        tag = 1
    if len(line.split('\t')) > 1:
        tag = 0
print('num of sentences which dont have trigger:', count)
print('num of all sentences:', all)
