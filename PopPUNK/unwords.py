import json
import gzip
import random
import string
import pkg_resources

# Based on a simple interpretation of https://simple.wikipedia.org/wiki/Syllable
def gen_unword(unique=True):
    # Download from https://github.com/dwyl/english-words/raw/master/words_dictionary.json
    word_list = pkg_resources.resource_stream(__name__, 'data/words_dictionary.json.gz')
    with gzip.open(word_list, 'rb') as word_list:
        real_words = json.load(word_list)

    vowels = ["a", "e", "i", "o", "u"]
    trouble = ["q", "x", "y"]
    consonants = set(string.ascii_lowercase) - set(vowels) - set(trouble)

    vowel = lambda: random.sample(vowels, 1)
    consonant = lambda: random.sample(consonants, 1)
    cv = lambda: consonant() + vowel()
    cvc = lambda: cv() + consonant()
    syllable = lambda: random.sample([vowel, cv, cvc], 1)

    returned_words = set()
    # Iterator loop
    while True:
      # Retry loop
      while True:
          word = ""
          for i in range(random.randint(2, 3)):
              word += "".join(syllable()[0]())
          if word not in real_words and (not unique or word not in returned_words):
              returned_words.add(word)
              break
      yield word
