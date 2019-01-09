import jieba
from bs4 import BeautifulSoup
from collections import defaultdict

# use combination to create data in sentence format
DATA_PATH = '../processed_data/combination.xml'
SAVE_PATH = '../processed_data/predata.txt'
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    soup = BeautifulSoup(f.read(), 'lxml')

docs = soup.find_all('doc')
# events = soup.find_all('event')

save_file = open(SAVE_PATH, 'w', encoding='utf-8')
for doc in docs:
    events = doc.find_all('event')
    save_file.write('id=' + doc.get('id') + '\n')

    sentences = {}
    trigger_idx = defaultdict(dict)

    for event in events:
        eventType = event.get('type')
        eventSubType = event.get('subtype')
        event_mentions = event.find_all('event_mention')

        for event_mention in event_mentions:
            ldc_scope = event.find('ldc_scope')
            ldc_scope_start = int(ldc_scope.find('charseq').get('start'))
            ldc_scope_end = int(ldc_scope.find('charseq').get('end'))
            ldc_scope_text = ldc_scope.get_text().strip()  # .replace('\n', '').replace(' ', '')

            # idx_spaces = set()
            # for i, w in enumerate(ldc_scope_text):
            #     if w == ' ':# and w == '\n':
            #         idx_spaces.add(i)
            #
            # ldc_scope_text = ldc_scope_text.replace('\n', '').replace(' ', '')

            if ldc_scope_start not in sentences:
                sentences[ldc_scope_start] = ldc_scope_text

            anchor = event.find('anchor')
            anchor_start = int(anchor.find('charseq').get('start'))
            anchor_end = int(anchor.find('charseq').get('end'))
            anchor_word = anchor.get_text()  # .replace('\n', '').replace(' ', '')

            trigger_start = anchor_start - ldc_scope_start
            trigger_end = anchor_end - ldc_scope_start + 1

            # count = 0
            # for i in idx_spaces:
            #     if i <= trigger_start:
            #         count += 1
            # trigger_start -= count

            trigger_idx[ldc_scope_start][trigger_start] = eventSubType

            # print(eventType, eventSubType, ldc_scope_text, anchor_word)
            # print(ldc_scope_start, ldc_scope_end)
            # print(anchor_start, anchor_end)
            print(ldc_scope_text[trigger_start: trigger_end], anchor_word.replace('\n', '').replace(' ', ''))
            # break
            # continue
            break
    print(trigger_idx)

    for i, sent in sentences.items():
        seg = list(jieba.cut(sent))
        idx = 0
        for s in seg:
            flag = 0
            for key in trigger_idx[i]:
                if idx <= key and key < idx + len(s):
                    # if idx == key:
                    if s != '\n' and s != ' ':
                        save_file.write(s + '\t' + trigger_idx[i][key] + '\n')
                        flag = 1
            if not flag and s != '\n' and s != ' ':
                save_file.write(s + '\n')
            idx += len(s)
        save_file.write('\n')

        # seg = list(jieba.cut(ldc_scope_text))
        # # print(' '.join(seg))
        # idx = 0
        #
        # # save_file.write(anchor_word.strip() + '\n')
        # for s in seg:
        #     if trigger_start >= idx and trigger_start < idx + len(s):
        #         print(s, eventType, eventSubType)
        #         if s != '\n' and s != ' ':
        #             save_file.write(s + '\t' + eventType + '\t' + eventSubType + '\n')
        #     else:
        #         print(s)
        #         if s != '\n' and s != ' ':
        #             save_file.write(s + '\n')
        #     idx += len(s)
        # save_file.write('\n')
        # break

save_file.close()
