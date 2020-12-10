import model
import nltk
from nltk.corpus import PlaintextCorpusReader
import numpy as np

###returns the difference between two sequences using the Levenshtein - Demerau distance 
###i.e the minimum number of single-character edits such as insertions, deletions, transpositions or substitutions 
def editDistance(s1, s2):
    row_size = len(s1) + 1
    col_size = len(s2) + 1
    matrix = np.zeros((row_size, col_size))
    for i in range(row_size):
        matrix[i, 0] = i
    for j in range(col_size):
        matrix[0, j] = j
    for i in range(1, row_size):
        for j in range(1, col_size):
            if s1[i - 1] == s2[j - 1]:
                diff = 0
            else:
                diff = 1
            matrix[i, j] = min(matrix[i - 1, j - 1] + diff,
                               matrix[i, j - 1] + 1,
                               matrix[i - 1, j] + 1)
            if i and j and s1[i - 2] == s2[j - 1] and s1[i - 1] == s2[j - 2]:
                matrix[i, j] = min(matrix[i, j], matrix[i - 2, j - 2] + 1)

    return matrix[-1, -1]

###returns weight of the single-character edit operation
def operationWeight(a, b):

    if a == None and len(b) == 1:
        return 3.0          # insertion
    elif b == None and len(a) == 1:
        return 3.0          # deletion
    elif a == b and len(a) == 1:
        return 0.0          # identity
    elif len(a) == 1 and len(b) == 1:
        return 2.5          # substitution
    elif len(a) == 2 and len(b)==2 and a[0] == b[1] and a[1] == b[0]:
        return 2.25         # transposition
    else:
        print("Wrong parameters ({},{}) of primitiveWeight call encountered!".format(a,b))

###returns the cost of all operations required to change one word into the other using the operationWeight function
def editWeight(s1, s2):
    row_size = len(s1) + 1
    col_size = len(s2) + 1
    matrix = np.zeros((row_size, col_size))
    for i in range(1, row_size):
        matrix[i, 0] = matrix[i - 1, 0] + operationWeight(s1[i - 1], None)
    for j in range(1, col_size):
        matrix[0, j] = matrix[0, j - 1] + operationWeight(None, s2[j - 1])
    for i in range(1, row_size):
        for j in range(1, col_size):
            matrix[i, j] = min(matrix[i - 1, j - 1] + operationWeight(s1[i - 1], s2[j - 1]),
                               matrix[i, j - 1] + operationWeight(None, s2[j - 1]),
                               matrix[i - 1, j] + operationWeight(s1[i - 1], None))
            if i and j and s1[i - 2] == s2[j - 1] and s1[i - 1] == s2[j - 2]:
                matrix[i, j] = min(matrix[i, j],
                                   matrix[i - 2, j - 2] + operationWeight(s1[i - 2] + s1[i - 1], s2[j - 2] + s2[j - 1]))

    return matrix[-1, -1]

###returns list of generated edits of some given query with caluclated Levenshtein-Demerau distance equal to one (dist == 1) 
###the function uses the alphabet given in in package model
def generateEdits(q):
    fst_half, snd_half = [], []
    for i in range(len(q) + 1):
        fst_half += [q[:i]]
        snd_half += [q[i:]]
    insert_edits = [fst_half_el + letter + snd_half_el
                    for fst_half_el, snd_half_el in zip(fst_half, snd_half)
                    for letter in model.alphabet]
    delete_edits = [fst_half_el + snd_half_el[1:]
                    for fst_half_el, snd_half_el in zip(fst_half, snd_half) if snd_half_el]
    transpose_edits = [fst_half_el + snd_half_el[1] + snd_half_el[0] + snd_half_el[2:]
                       for fst_half_el, snd_half_el in zip(fst_half, snd_half) if len(snd_half_el) >= 2]
    replaces_edits = [fst_half_el + letter + snd_half_el[1:]
                      for fst_half_el, snd_half_el in zip(fst_half, snd_half) if snd_half_el
                      for letter in model.alphabet]
    return [x for x in set(insert_edits + delete_edits + transpose_edits + replaces_edits) if x != q]
  



###returns set of pairs of candidates and their logarithm probability which is minus weigth of all operations 
def generateCandidates(query,dictionary):
    def allWordsInDictionary(q): ###returns if all words of the query are in the dictionary
        return all(w in dictionary for w in q.split())
    edits = generateEdits(query)
    return set([(candidate, -editWeight(query, candidate))
                for edit in edits for candidate in generateEdits(edit)
                if allWordsInDictionary(candidate) and editDistance(query, candidate) <= 2])


###returns the best of all candidates generated with function generateCandidates by their probability using both probabiity from
###the statistical stochastic Markov model implemented in the package model and the logarithm probability            
def correctSpelling(r, model, mu = 1.0, alpha = 0.9):

    def getScore(q,logEditProb):
        return pow(2, logEditProb)*pow((pow(2, model.sentenceLogProbability(q.split(), alpha))), mu)
        
    candidates = generateCandidates(r, model.kgrams[tuple()])
    prob_candidates = [(candidate[0], getScore(candidate[0], candidate[1])) for candidate in candidates]
    return (sorted(prob_candidates, key=lambda x: x[1], reverse=True))[0][0]

