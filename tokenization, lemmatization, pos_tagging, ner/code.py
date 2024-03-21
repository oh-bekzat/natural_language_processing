import re
import stanza
from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2


# TASK 1 begin


path = 'kk_ktb-ud-test.conllu'
with open(path, 'r') as file:
    ktb_contents = file.read()

ktb_sentence_sections = [sentence_section.strip() for sentence_section in ktb_contents.strip().split('\n\n')]

# creating an array of arrays, where each array is a sentence with words as objects
ktb_sentences_data = []
range_pattern = re.compile(r'^\d+-\d+(\t.*)?$') # some start as 4-5, 5-6, etc., indicating connection between lines, irrelevant for comparison
for section in ktb_sentence_sections:
    sentence_data = []
    for line in section.split('\n'):
        if not line.startswith('#') and not range_pattern.match(line):  # excluding lines starting with '#' and matching the pattern
            parts = line.split('\t')
            word_info = {'lemma': parts[2], 'pos': parts[3]}
            sentence_data.append(word_info)
    ktb_sentences_data.append(sentence_data)

# taking only the sentences from the ktb data for stanza
ktb_lines = ktb_contents.split('\n')
ktb_sentences = [line.split('# text = ')[-1] for line in ktb_lines if line.startswith('# text =')]
ktb_tokenized = "\n\n".join(ktb_sentences)

nlp = stanza.Pipeline(lang = 'kk', processors = 'tokenize, lemma, pos', tokenize_no_ssplit = True)
doc = nlp(ktb_tokenized)

# creating an array of arrays, where each array is a sentence with words as objects, correspond to the ktb data
stanza_sentences_data = []
for sentence in doc.sentences:
    sentence_data = []
    for word in sentence.words:
        word_info = {'lemma': word.lemma,'pos': word.upos}
        sentence_data.append(word_info)
    stanza_sentences_data.append(sentence_data)

# for i, stanza_sentence_data in enumerate(stanza_sentences_data):
#     for j, stanza_word in enumerate(stanza_sentence_data):
#         if j <= len(ktb_sentences_data[i]):
#             print(stanza_word['lemma'], ktb_sentences_data[i][j]['lemma'])
#             print(stanza_word['pos'], ktb_sentences_data[i][j]['pos'])
#     print('\n')

# comparing the two data sets
matching_sentences_count = sentences_count = len(stanza_sentences_data)
non_matching_lemma_count = non_matching_pos_count = total_lemma_count = total_pos_count = 0
for i, stanza_sentence_data in enumerate(stanza_sentences_data):
    if len(stanza_sentence_data) != len(ktb_sentences_data[i]):
        matching_sentences_count -= 1
        # print(stanza_sentence_data)
        # print(ktb_sentences_data[i])
        # print('\n')
    else:
        total_lemma_count += len(stanza_sentence_data)
        total_pos_count += len(stanza_sentence_data)
        for j, stanza_word in enumerate(stanza_sentence_data):
            if stanza_word['lemma'] != ktb_sentences_data[i][j]['lemma']:
                non_matching_lemma_count += 1
            if stanza_word['pos'] != ktb_sentences_data[i][j]['pos']:
                non_matching_pos_count += 1

# print('Matching sentences count =', matching_sentences_count)
# print('Non-matching sentences count =', sentences_count - matching_sentences_count)
# print('Total sentences count =', sentences_count)
# # calculating accuracies
sentences_accuracy = (matching_sentences_count / sentences_count) * 100
lemmatization_accuracy = ((total_lemma_count - non_matching_lemma_count) / total_lemma_count) * 100
pos_tagging_accuracy = ((total_pos_count - non_matching_pos_count) / total_pos_count) * 100

print('Sentence accuracy =', sentences_accuracy)
print('Lemmatization accuracy =', lemmatization_accuracy)
print('POS tagging accuracy =', pos_tagging_accuracy)


# TASK 1 end


# TASK 2 begin


path = 'IOB2_test.txt'
with open(path, 'r') as file:
    kaznerd_contents = file.read()

# print(kaznerd_contents)

# preparing the data for stanza and true labels for comparison
sentences = [sentence.strip() for sentence in kaznerd_contents.split('\n\n') if sentence.strip()]
kaznerd_ner = [] # ner tags, list of lists
kaznerd_text = []
for sentence in sentences:
    sentence_kaznerd_ner = []
    words = sentence.split('\n')
    word_only = []
    for word in words:
        word_only.append(word.split(' ')[0]) # collecting only the words
        sentence_kaznerd_ner.append(word.split(' ')[-1]) # collecting only the NER tags
    kaznerd_sentence = ' '.join(word_only)
    kaznerd_text.append(kaznerd_sentence)
    kaznerd_ner.append(sentence_kaznerd_ner)
kaznerd_text = '\n\n'.join(kaznerd_text) # tokenized text for stanza

# print(kaznerd_text)

nlp = stanza.Pipeline(lang = 'kk', processors = 'tokenize, ner', tokenize_pretokenized = True)
doc = nlp(kaznerd_text)

# preparing ner tags from stanza as predicted labels, list of lists
stanza_ner = []
for sentence in doc.sentences:
    sentence_ner = []
    for token in sentence.tokens:
        if token.ner.startswith('S'):
            sentence_ner.append('B' + token.ner[1:])
        elif token.ner.startswith('E'):
            sentence_ner.append('I' + token.ner[1:])
        else: sentence_ner.append(token.ner)
    stanza_ner.append(sentence_ner)

# print(stanza_ner)

# evaluation of ner result, kaznerd is true labels, stanza is predicted labels
accuracy = accuracy_score(kaznerd_ner, stanza_ner)
report = classification_report(kaznerd_ner, stanza_ner, zero_division = 0, mode = 'strict', scheme = IOB2)
print('Accuracy =', accuracy)
print(report)

# for i, s in enumerate(kaznerd_ner):
#     for j, m in enumerate(s):
#         print(j, m, stanza_ner[i][j])
#     print('\n')


# TASK 2 end


# TASK 3 begin


text = 'Шыңғысхан құрған монғол хандығының ғұмыры екі жүз жылға жетпеді.\n\nБір кездегі ұлы көшпелі мемлекет — Қарақұрым ордасы Құбылайдың тұсында Пекинге көшісімен-ақ монғол хандығы делінуден қалды.\n\nҚұбылайдан кейінгі Қытай боғдыхандары енді өздерін Шыңғыс мұрагерлері санап, монғолдың атамекен көне қонысы түгіл, «Бар әлемді тітіретуші» жирен сақалды ханның жаулап алған жерлерін де бауырларына басқысы келді.\n\nБұлар енді бір кезде ұлы Қытай империясын Шыңғысханның күшпен жаулап алғанын, оның көп шаһарларын тып-типыл етіп қиратып, егістік даласын малға жайылым еткісі келгенін ұмытты.\n\nАл монғол жеріндегі ұлы Қарақұрым хандығы да бөлшектене бастады.\n\nӨзара қырқыс, жанжал бір жағынан, күнгей үрдісінде пайда болған манчжур хандарының ұзақ жылғы ұрыстары екінші жағынан берекесін алып, бұлардың бұрынғыдай іргелі ел болып отыруына мүмкіндік бермеді.\n\nОның үстіне негізгі кәсібі мал бағу болған, әр аулы әр бөлек қонған монғол шонжарларына қыс — қыстау, жаз — жайлау жетпей, елге қоныс, малға өріс тапшылығы тағы бір пәле болды.\n\nӘсіресе батыс монғол тайпалары — Чорас, Ойрат, Торғауыт, Төлеуіт рулары Қытай боғдыхандарының тегеурініне шыдай алмай атамекен қоныстарын тастап, жер іздеп босып кеткен.\n\nБір бөлегі Сібір жеріне, қалғаны Ертіс бойына, Тарбағатай тауына қарай ойысты.\n\nҚалмақ аталған бір бөлегі жер іздеп, көше-көше тіпті Еділдің төменгі сағасына өтіп кетіп, Айдархан (Астрахань) маңайында көшпелі аймақ боп тұрып қалды.'
nlp = stanza.Pipeline(lang = 'kk', processors = 'tokenize, lemma, pos, ner', tokenize_no_ssplit = True)
doc = nlp(text)

# print(doc)

# creating an array of arrays, where each array is a sentence with words as objects, correspond to the ktb data
sentences_data = []
for sentence in doc.sentences:
    sentence_data = []
    for word in sentence.words:
        word_info = {'lemma': word.lemma,'pos': word.upos}
        sentence_data.append(word_info)
    sentences_data.append(sentence_data)

for sentence_data in sentences_data:
    for word in sentence_data:
        print(word['lemma'], word['pos'])
    print('\n')

for sentence in doc.sentences:
    for entity in sentence.ents:
        print(f"token: {entity.text}, ner: {entity.type}")


# TASK 3 end