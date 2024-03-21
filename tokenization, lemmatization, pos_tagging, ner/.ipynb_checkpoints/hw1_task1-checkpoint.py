import re
import stanza

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
# sentences_accuracy = (matching_sentences_count / sentences_count) * 100
# # lemmatization_accuracy = ((total_lemma_count - non_matching_lemma_count) / total_lemma_count) * 100
# # pos_tagging_accuracy = ((total_pos_count - non_matching_pos_count) / total_pos_count) * 100

# # print('Sentence accuracy =', sentences_accuracy)
# # print('Lemmatization accuracy =', lemmatization_accuracy)
# print('POS tagging accuracy =', pos_tagging_accuracy)