import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.probability import FreqDist
from nltk.tokenize.treebank import TreebankWordDetokenizer
import networkx as nx

def outputsumm(input_text, num_sentences=5):
    sentences = sent_tokenize(input_text)
    stop_words = set(stopwords.words("english"))
    words = [word.lower() for word in word_tokenize(input_text) if word.isalnum() and word.lower() not in stop_words]
    # word_frequencies = FreqDist(words)
    graph = nx.Graph()
    for sentence1 in sentences:
        for sentence2 in sentences:
            if sentence1 != sentence2:
                overlap = set(word_tokenize(sentence1)).intersection(set(word_tokenize(sentence2)))
                weight = len(overlap) / (len(set(word_tokenize(sentence1))) + len(set(word_tokenize(sentence2))))
                graph.add_edge(sentence1, sentence2, weight=weight)
    scores = nx.pagerank(graph)
    ranked_sentences = sorted(((scores[sentence], sentence) for sentence in sentences), reverse=True)
    selected_sentences = [sentence for score, sentence in ranked_sentences[:num_sentences]]
    summary = TreebankWordDetokenizer().detokenize(selected_sentences)
    return summary
