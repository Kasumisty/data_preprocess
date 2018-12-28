import jieba
from bs4 import BeautifulSoup

DATA_PATH = '../processed_data/combination.xml'
SAVE_PATH = '../processed_data/predata_one_mention.txt'
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    soup = BeautifulSoup(f.read(), 'lxml')
events = soup.find_all('event')

# id = 1
save_file = open(SAVE_PATH, 'w', encoding='utf-8')

for event in events:
    eventType = event.get('type')
    eventSubType = event.get('subtype')
    event_mentions = event.find_all('event_mention')[0]

    for event_mention in event_mentions:
        ldc_scope = event.find('ldc_scope')
        ldc_scope_start = int(ldc_scope.find('charseq').get('start'))
        ldc_scope_end = int(ldc_scope.find('charseq').get('end'))
        ldc_scope_text = ldc_scope.get_text().strip()  # .replace('\n', '').replace(' ', '')

        anchor = event.find('anchor')
        anchor_start = int(anchor.find('charseq').get('start'))
        anchor_end = int(anchor.find('charseq').get('end'))
        anchor_word = anchor.get_text()  # .replace('\n', '').replace(' ', '')

        trigger_start = anchor_start - ldc_scope_start
        trigger_end = anchor_end - ldc_scope_start + 1

        # print(eventType, eventSubType, ldc_scope_text, anchor_word)
        # print(ldc_scope_start, ldc_scope_end)
        # print(anchor_start, anchor_end)
        # print(ldc_scope_text[trigger_start: trigger_end])

        # break
        # continue

        seg = list(jieba.cut(ldc_scope_text))
        # print(' '.join(seg))
        idx = 0

        save_file.write(anchor_word.strip() + '\n')
        for s in seg:
            # idx += len(s)
            if idx == trigger_start:
                print(s, eventType, eventSubType)
                if s != '\n' and s != ' ':
                    save_file.write(s + '\t' + eventType + '\t' + eventSubType + '\n')
            else:
                print(s)
                if s != '\n' and s != ' ':
                    save_file.write(s + '\n')
            idx += len(s)
        save_file.write('\n')
    # break
