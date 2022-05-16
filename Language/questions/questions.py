import os

import nltk
import sys
import string
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    files = dict()
    for subdir in os.listdir(directory):
        file_path = os.path.join(directory, subdir)
        file = open(file_path, "r") #open in read mode
        files[subdir] = file
    return files

def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    tokens = nltk.word_tokenize(document)
    words = []
    for token in tokens:
        if token not in string.punctuation and token not in nltk.corpus.stopwords.words("english"):
            words.append(token.lower())
    return words

def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    freq = dict([])
    for doc in documents.keys():
        for word in documents[doc]:
            if doc not in freq[word]: #
                freq[word].append(doc)
    idfs = dict()
    for word in freq.keys():
        idfs[word] = math.log(len(documents)/len(freq[word]))
    return idfs

def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    tf_idf = dict()
    for word in query:
        for file in files.keys():
            if word in files[file]:
                tf = files[file].count(word)
                idf = idfs[word]
                if file not in tf_idf.keys():
                    tf_idf[file] = tf*idf
                else:
                    tf_idf[file] += tf*idf
    sorted_tf_idf = sorted(tf_idf.items(), key=lambda x:x[1], reverse=True)
    final = sorted_tf_idf[:n]
    return final

def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    def QRTD(query, sentence):
        length = len(sentence)
        count = 0
        for word in query:
            if word in sentence:
                count += 1
        return count/float(length)

    vals = dict()
    for word in query:
        for sentence in sentences.keys():
            if word in sentences[sentence]:
                idf = idfs[word]
                if sentence not in vals.keys():
                    vals[sentence] = (sentences[sentence], idf, QRTD(query, sentences[sentence]))
                else:
                    vals[sentence][1] += idf
    sorted_vals = sorted(vals, key= lambda x: (x[1], x[2]), reverse=True)
    final = sorted_vals[:n][0]
    return final

if __name__ == "__main__":
    main()
