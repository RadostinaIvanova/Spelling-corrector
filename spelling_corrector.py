#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2020/2021
#############################################################################

### Домашно задание 1
###
### За да работи програмата, трябва да се свали корпус от публицистични текстове за Югоизточна Европа,
### предоставен за некомерсиално ползване от Института за български език - БАН
###
### Корпусът може да бъде свален от:
### Отидете на http://dcl.bas.bg/BulNC-registration/#feeds/page/2
### И Изберете:
###
### Корпус с новини
### Корпус от публицистични текстове за Югоизточна Европа.
### 27.07.2012 Български
###    35337  7.9M
###
### http://dcl.bas.bg/BulNC-registration/dl.php?dl=feeds/JOURNALISM.BG.zip
###
### Архивът трябва да се разархивира в директорията, в която е програмата.
###
### Преди да се стартира програмата, е необходимо да се активира съответната среда с командата:
### conda activate tii
###
### Ако все още нямате създадена среда, прочетете файла README.txt за инструкции

import model
import nltk
from nltk.corpus import PlaintextCorpusReader
import numpy as np

def editDistance(s1, s2):
    ### Функцията намира разстоянието на Левенщайн-Дамерау между два низа
    ### Вход:
    ###     низ: s1
    ###     низ: s2
    ### Изход: минималният брой на елементарните операции (вмъкване, изтриване, субституция и транспозиция на символи) необходими да се получи единият низ от другия
    #############################################################################
    ### Начало на Вашия код. На мястото на pass се очакват 10-25 реда
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
    ### Край на Вашия код
    #############################################################################


def operationWeight(a, b):
    ### Функцията operationWeight връща теглото на дадена елементарна операция
    ### Тук сме реализирали функцията съвсем просто -- връщат се фиксирани тегла, които не зависят от конкретните символи. При наличие на статистика за честотата на грешките, тези тегла следва да се заменят със съответни тегла, получени след оценка на вероятността за съответната грешка, като се използва принципът за максимално правдоподобие
    ### Вход:
    ###     низ: a
    ###     низ: b
    ### Изход: теглото за операцията
    ### ВАЖНО: При изтриване и вмъкване се предполага, че празният низ е представен с None
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

def editWeight(s1, s2):
    ### Функцията editWeight намира теглото между два низа. За намиране на елеметарните тегла следва да се извиква функцията operationWeight
    ### Вход:
    ###     низ s1
    ###     низ s2
    ### Изход: минималното тегло за подравняване, за да се получи от единия низ другият
    #############################################################################
    ### Начало на Вашия код. На мястото на pass се очакват 10-25 реда
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

    ### Край на Вашия код
    #############################################################################


def generateEdits(q):
    ### Помощната функция generateEdits по зададена заявка генерира всички възможни редакции на разстояние 1 от тази заявка
    ### Вход:
    ###     низ: q, представящ заявката
    ### Изход: списък от низове на разстояние 1 по Левенщайн-Дамерау от заявката
    ### Забележка: в тази функция вероятно ще трябва да използвате азбука, която е дефинирана в model.alphabet
    #############################################################################
    ### Начало на Вашия код. На мястото на pass се очакват 10-15 реда
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
    ### Край на Вашия код
    #############################################################################




def generateCandidates(query,dictionary):
    ### Функцията започва от заявката query и генерира всички низове НА РАЗСТОЯНИЕ <= 2, за да се получат кандидатите за корекция. Връщат се единствено кандидати, всички думи на които са в речника dictionary.
    ### Вход:
    ###     низ: query
    ###     речник: dictionary
    ### Изход: списък от двойки (candidate, candidateEditLogProbability), където candidate е низ-кандидат, а candidateEditLogProbability е логаритъм от вероятността за редакция, тоест минус теглото.
    def allWordsInDictionary(q):
        ### Помощната функция връща истина, ако всички думи в заявката са в речника
        return all(w in dictionary for w in q.split())
    #############################################################################
    ### Начало на Вашия код. На мястото на pass се очакват 10-15 реда
    if allWordsInDictionary(query):
        return [(query, 0.0)]
    edits = generateEdits(query)
    return set([(candidate, -editWeight(query, candidate))
                for edit in edits for candidate in generateEdits(edit)
                if allWordsInDictionary(candidate) and editDistance(query, candidate) <= 2])
    ### Край на Вашия код
    #############################################################################


def correctSpelling(r, model, mu = 1.0, alpha = 0.9):
    ### Функцията комбинира езиковия модел model с кандидатите за корекция, генерирани от generateCandidates, за намиране на най-вероятната желана заявка от дадената оригинална заявка query. Функцията, която генерира кандидати, връща и вероятността за редактиране
    ### Вход:
    ###    низ: r, представящ заявката
    ###    езиков модел: model
    ###    число: mu -- тегло на езиковия модел
    ###    число: alpha -- коефициент за интерполация на езиковия модел
    ### Изход: най-вероятната заявка

    def getScore(q,logEditProb):
        ### Функцията използва езиковия модел и вероятността за редакцията logEditProb за изчисляване на оценка за кандидата q. Използва mu като степен на тежест за Pr[q].
        ### Вход:
        ###     низ: q -- кандидат заявка
        ###     число: logEditProb -- логаритъм от вероятността за редакция на дадената заявка (т.е. log Pr[r|q], където r е оригиналната заявка)
        ### Изход: логаритъм от вероятността за кандидат заявката
        #############################################################################
        #### Начало на Вашия код. На мястото на pass се очакват 1-3 реда
        return pow(2, logEditProb)*pow((pow(2, model.sentenceLogProbability(q.split(), alpha))), mu)
        #### Край на Вашия код
        #############################################################################
        ####
    ###
    #############################################################################
    #### Начало на Вашия код за основното тяло на функцията correctSpelling. На мястото на pass се очакват 2-5 реда
    candidates = generateCandidates(r, model.kgrams[tuple()])
    prob_candidates = [(candidate[0], getScore(candidate[0], candidate[1])) for candidate in candidates]
    return (sorted(prob_candidates, key=lambda x: x[1], reverse=True))[0][0]
    #### Край на Вашия код
    #############################################################################

corpus_root = 'JOURNALISM.BG/C-MassMedia'
myCorpus = PlaintextCorpusReader(corpus_root, '.*\.txt')
fullSentCorpus = [ [model.startToken] + [w.lower() for w in sent] + [model.endToken] for sent in myCorpus.sents()]
print('Готово.')

print('Трениране на Марковски езиков модел...')
M2 = model.MarkovModel(fullSentCorpus, 2)
print('Готово.')

