import nltk
from nltk.corpus import words
from nltk.corpus import wordnet as wn

# Ensure that all necessary NLTK data and corpora are downloaded
nltk.download("words")
nltk.download("wordnet")

# Load the initial word list from the 'words' corpus
word_list = set(words.words())

# Add all lemmas from WordNet to the word list using a list comprehension
wordnet_lemmas = {
    lemma.name().replace("_", " ")
    for synset in wn.all_synsets()
    for lemma in synset.lemmas()
}
word_list.update(wordnet_lemmas)


def is_valid_word(word):
    # Check if the word exists in WordNet
    return word in word_list


def fix_spacing(text: str):
    # Split the text into words
    words = text.split()
    i = 0
    while i < len(words) - 1:
        # Try merging the current word with the next one
        merged_word = words[i] + words[i + 1]
        merged_is_valid = is_valid_word(merged_word.lower())
        first_is_valid = is_valid_word(words[i].lower())
        second_is_valid = is_valid_word(words[i + 1].lower())
        # print(merged_word, merged_is_valid)
        # print(words[i], first_is_valid)
        # print(words[i + 1], second_is_valid)
        # Check if the merged word is in the word list
        if merged_is_valid and (not first_is_valid or not second_is_valid):
            # If it is, replace the current word with the merged one and delete the next word
            words[i] = merged_word
            del words[i + 1]
        else:
            # Otherwise, just move to the next word
            i += 1
    return " ".join(words)
