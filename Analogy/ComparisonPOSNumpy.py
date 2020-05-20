import numpy
from pycorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP('http://localhost:9000')
import Analogy.NLP_POSPair
import os

class MainComparison:

    def __fetchNSubj(annotatedObject, text):
        try:
            iter = (len(annotatedObject['sentences'][0]['tokens']))
        
            for i in range(iter-1):
                if(annotatedObject['sentences'][0]['basicDependencies'][i]['dep'] == "nsubj"):
                    nsubj = (annotatedObject['sentences'][0]['basicDependencies'][i]['dependentGloss'])
                    return nsubj
            return None
    
        except Exception as ex:
            print(ex)
    
    def __pos(index, annotatedObject, word):
            #fetching part-of-speech of specific word from annotated parsed sentence (Further on using pos = part-of-speech)
            try:
                i = 0
                while True:
                    #Finding pos of word from annotated object
                    if(annotatedObject['sentences'][index]['tokens'][i]['originalText'] == word): 
                        partOfSpeech = annotatedObject['sentences'][index]['tokens'][i]['pos']
                        break
                    else:
                        i = i + 1     
                
                #pos notations #Converting it into 8 major part-of-speech           
                noun = ['NN', 'NNS', 'NNP', 'NNPS', 'POS']
                pronoun = ['PRP', 'PRP$', 'WP', 'WP$']
                adjective = ['JJ', 'JJR', 'JJS', 'CD', 'DT', 'EX', 'FW', 'LS', 'PDT', 'RP', 'WDT']                
                verb = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'MD']
                adverb = ['RB', 'RBR', 'RBS', 'WRB']
                preposition = ['IN', 'TO']
                conjunction = ['CC']
                interjection = ['UH']
                others = ['SYM']
                notations = [",",".","?",";",":","<",">","/","'",'"','!','&']
        
                if(partOfSpeech in noun):
                    return 'Noun'
                if(partOfSpeech in pronoun):
                    return 'Pronoun'
                if(partOfSpeech in adjective):
                    return 'Adjective'
                if(partOfSpeech in verb):
                    return 'Verb'
                if(partOfSpeech in adverb):
                    return 'Adverb'
                if(partOfSpeech in preposition):
                    return 'Preposition'
                if(partOfSpeech in conjunction):
                    return 'Conjunction'
                if(partOfSpeech in interjection):
                    return 'Interjection'
                if(partOfSpeech in others):
                    return 'null'
                if(partOfSpeech in notations):
                    return 'null'
                if not partOfSpeech: #if pos is blank or error
                    return 'null'
        
            except:
                return None

    def __fetchInputIndex(array):
        try:
            #Finding the index to insert values in array
            index = 0
            while True:
                if(array[index, 0] == None):
                    return index
                else:
                    index += 1

        except Exception as ae:
            print(ae)

    def __selectBaseWord(sentences, ndSentences = numpy.empty((2000000, 3), dtype = object), baseWords = numpy.empty((2000000, 2), dtype = object)): #Input as list of text
        try:
            #Fetching the index to insert values
            index = MainComparison.__fetchInputIndex(ndSentences)
            index2 = MainComparison.__fetchInputIndex(baseWords)

            for sentence in sentences:
                #If input is empty
                if(len(sentences) == 0):
                    print("Input is empty")
                    return None
            
                try:
                    annotatedObject = nlp.annotate(sentence, properties = {
                        'annotators' : 'depparse',
                        'outputFormat' : 'json'
                    })
                except Exception as ae:
                    print(ae)
                    continue
        
                nsubj = MainComparison.__fetchNSubj(annotatedObject, sentence)
                if(nsubj == None):
                    continue
        
                posPairs = Analogy.NLP_POSPair.NLP.WordPairsWithValues(sentence, annotatedObject)
                if(posPairs is None):
                    continue
                if(isinstance(posPairs, list) == True):
                    if(None in posPairs):
                        continue
            
                baseWordPairs = []
                for i in posPairs:
                    for j in i:
                        if(j.word == nsubj):
                            if(j.posOfValue == 'Noun'):
                                if(j.word.lower() == j.value.lower()):
                                    continue
                                baseWordPairs.append(j.word.lower() + " " + j.value.lower())
                            
                pairString = ''
                for i in posPairs:
                    for j in i:
                        if(pairString == ''):
                            pairString = j.word + " " + j.value
                        else:
                            pairString = pairString + "," + j.word + " " + j.value
        
                #Store in two numpy arrays 1. ndSentences 2. baseWords
                ndSentences[index] = [sentence.lower(), nsubj.lower(), pairString.lower()]
                index += 1

                for ind, i in enumerate(baseWordPairs):
                    pair = i.split()
                    baseWords[index2] = [pair[0].lower(), pair[1].lower()]
                    index2 += 1
                    
            return ndSentences, baseWords

        except Exception as ae:
            print(ae)

    def __fetchBaseWord(baseWords, word):
        try:
            baseWordsList = numpy.where(baseWords[:,0] == word.lower())
        
            temp = baseWords[baseWordsList]
            baseWord = []
        
            for i in range(numpy.size(baseWords[baseWordsList], 0)):
                baseWord.append(temp[i, 1])
            baseWord = list(set(baseWord))
            return baseWord
    
        except Exception as ae:
            print(ae)

    def __fetchWordfromBaseWord(baseWords, word):
        try:
            WordsList = numpy.where(baseWords[:,1] == word.lower())
            temp = baseWords[WordsList]
            Words = []
        
            for i in range(numpy.size(baseWords[WordsList], 0)):
                Words.append(temp[i, 0])
            Words = list(set(Words))
            return Words

        except Exception as ae:
            print(ae)

    def __diffValidity(tempData1, tempData2, word1, word2, diff1, diff2):
        try:
            if(len(tempData1) == 0 or len(tempData2) == 0):
                return True

            for pairs in tempData1:
                if(diff2 == pairs[1]):
                    return False
    
            for pairs in tempData2:
                if(diff1 == pairs[1]):
                    return False

        except Exception as ae:
            print(ae)

    def findComparison(model, word1, word2):
        try:
            word1 = word1.lower()
            word2 = word2.lower()
        
            result = []
            sim,diff = MainComparison.__comparison(model['ndSentences'], model['baseWords'], word1, word2)
        
            for i in sim:
                if(len(sim) == 0):
                    continue
                if(i[2] != None):
                    result.append(MainComparison.Compare(word1, word2, i[2], None, None, None))

            for i in diff:
                if(len(diff) == 0):
                    continue
                if(i[2] != None):
                    result.append(MainComparison.Compare(word1, word2, None, i[2], i[3], i[4]))

            if(len(result) > 0):
                return result
            else:
                return None

        except Exception as ae:
            print(ae)

    def __comparison(ndSentences, baseWords, word1, word2):
        try:
            sim = []
            diff = []
            length = numpy.size(ndSentences, 0)

            #If words are same
            if(word1 == word2):
                return sim, diff
        
            #Fetch data of both words
            sentenceData1 = MainComparison.__fetchSentencesData(ndSentences, word1)
            sentenceData2 = MainComparison.__fetchSentencesData(ndSentences, word2)
        
            #Fetch pairs data
            tempData1 = []
            tempData2 = []
            fetchedData1 = []
            fetchedData2 = []
        
            #Iterating through both sentences
            for i in sentenceData1:
                nsubj1 = i[1]
            
                if(nsubj1 == None):
                    continue

                sentence1Pairs = i[2].split(',')
            
                for pairs1 in sentence1Pairs:
                    flag = True
                    pair1 = pairs1.split(" ")
                    if(pair1[0].lower() == nsubj1.lower()):
                        for p in fetchedData1:
                            if(p == pair1[1]):
                                flag = False
                        if(flag == True):
                            baseWordsList = MainComparison.__fetchBaseWord(baseWords, pair1[1])
                            for m in baseWordsList:
                                tempData1.append([pair1[0], pair1[1], m])
                            fetchedData1.append(pair1[1])

            for j in sentenceData2:
                nsubj2 = j[1]
                
                if(nsubj1 == None):
                    continue
        
                sentence2Pairs = j[2].split(',')
    
                for pairs2 in sentence2Pairs:
                    flag = True
                    pair2 = pairs2.split(" ")
                    if(pair2[0].lower() == nsubj2.lower()):
                        for p in fetchedData2:
                            if(p == pair2[1]):
                                flag = False
                        if(flag == True):
                            baseWordsList = MainComparison.__fetchBaseWord(baseWords, pair2[1])
                            for m in baseWordsList:
                                tempData2.append([pair2[0], pair2[1], m])
                            fetchedData2.append(pair2[1])

            #For similiarity
            fetchedSimiliarity = []
            for k in tempData1:
                for l in tempData2:
                    if(k[1] == l[1]):
                        flag = True
                        for m in fetchedSimiliarity:
                            if(m == k[1]):
                                flag = False
                        if(flag == True):
                            sim.append([k[0], l[0], k[1]])
                            fetchedSimiliarity.append(k[1])

            #For difference
            for k in tempData1:
                for l in tempData2:
                    if(k[2] == l[2]):
                        #Check if its really a difference
                        if(MainComparison.__diffValidity(tempData1, tempData2, nsubj1, nsubj2, k[1], l[1]) == False):
                            continue
                        diff.append([k[0], l[0], k[1], l[1], k[2]])
                    
            return sim, diff

        except Exception as ae:
            print(ae)

    def __fetchSentencesData(ndSentences, word):
        try:
            sentencesData = numpy.where(ndSentences[:,1] == word.lower())
        
            temp = ndSentences[sentencesData]
        
            sentences = []
            for i in range(numpy.size(ndSentences[sentencesData], 0)):
                sentences.append(temp[i])
        
            return sentences
    
        except Exception as ae:
            print(ae)

    def __fetchBaseWordData(baseWords, word):
        try:
            baseWordsData = numpy.where(baseWords[:,0] == word.lower())
        
            temp = baseWords[baseWordsData]
            baseWord = []
        
            for i in range(numpy.size(baseWords[baseWordsData], 0)):
                baseWord.append(temp[i])
        
            return baseWord
    
        except Exception as ae:
            print(ae)

    class Compare(object):
            Word1 = ""
            Word2 = ""
            Similiarity = ""
            Difference1 = ""
            Difference2 = ""
            Hypernym = ""

            def __init__(self, Word1, Word2, Similiarity, Difference1, Difference2, Hypernym):
                self.Word1 = Word1
                self.Word2 = Word2
                self.Similiarity = Similiarity
                self.Difference1 = Difference1
                self.Difference2 = Difference2
                self.Hypernym = Hypernym
            
            def ObjectCreate(Word1, Word2, Similiarity, Difference1, Difference2, Hypernym):
                createdObject = Compare(Word1, Word2, Similiarity, Difference1, Difference2, Hypernym)
                return createdObject

    def findSentenceComparison(model, sentence1, sentence2):
        try:
            if(sentence1.lower() == sentence2.lower()):
                return None
        
            annotatedObject1 = nlp.annotate(sentence1, properties = {
                        'annotators' : 'depparse',
                        'outputFormat' : 'json'
                    })
            annotatedObject2 = nlp.annotate(sentence2, properties = {
                        'annotators' : 'depparse',
                        'outputFormat' : 'json'
                    })
        
            posPairs1 = Analogy.NLP_POSPair.NLP.WordPairsWithValues(sentence1, annotatedObject1)
            posPairs2 = Analogy.NLP_POSPair.NLP.WordPairsWithValues(sentence2, annotatedObject2)
        
            nsubj1 = MainComparison.__fetchNSubj(annotatedObject1, sentence1)
            nsubj2 = MainComparison.__fetchNSubj(annotatedObject2, sentence2)
        
            iter1 = (len(annotatedObject1['sentences'][0]['tokens']))
            iter2 = (len(annotatedObject2['sentences'][0]['tokens']))
        
            comparison = []
            doneWords1 = []
            doneWords2 = []
        
            if(nsubj1 != None and nsubj2 != None):
                comparison.append(MainComparison.findComparison(model, nsubj1, nsubj2))
                doneWords1.append(nsubj1)
                doneWords2.append(nsubj2)
        
            #Finding words with syntactic relation and their pos
            directRelation1 = []
            directRelation2 = []
            if(posPairs1 != None and posPairs2 != None):
                for i in posPairs1:
                    for j in i:
                        if(j.word == nsubj1):
                            directRelation1.append(j.value + " " + j.posOfValue)
                
                for i in posPairs2:
                    for j in i:
                        if(j.word == nsubj2):
                            directRelation2.append(j.value + " " + j.posOfValue)

            similiarityOfRelated = []
            for i in directRelation1:
                for j in directRelation2:
                    pair1 = i.split(" ")
                    pair2 = j.split(" ")
                    if(pair1[1] == pair2[1]):
                        if(MainComparison.__wordInList(pair1[0], doneWords1) == False and MainComparison.__wordInList(pair2[0], doneWords2) == False):
                            comparison.append(MainComparison.findComparison(model, pair1[0], pair2[0]))
                            doneWords1.append(pair1[0])
                            doneWords2.append(pair2[0])
                            break

            #What if nsubj not present, then matching words according to their pos
            sentenceTokens = []
            for i in range(iter1-1):
                for j in range(iter2-1):
                    word1 = annotatedObject1['sentences'][0]['tokens'][i]['originalText']
                    word2 = annotatedObject2['sentences'][0]['tokens'][j]['originalText']
                    if(MainComparison.__wordInList(word1, doneWords1) == False and MainComparison.__wordInList(word2, doneWords2) == False):
                        pos1 = MainComparison.__pos(0, annotatedObject1, word1)
                        pos2 = MainComparison.__pos(0, annotatedObject2, word2)
                        if(pos1 == pos2):
                            comparison.append(MainComparison.findComparison(model, word1, word2))
                            doneWords1.append(word1)
                            doneWords2.append(word2)
                            break
        
            #Matching remaining words
            if(iter1 < iter2 or iter1 == iter2):
                iter = iter1
            else:
                iter = iter2
            for i in range(iter):
                word1 = annotatedObject1['sentences'][0]['tokens'][i]['originalText']
                word2 = annotatedObject2['sentences'][0]['tokens'][i]['originalText']
                if(MainComparison.__wordInList(word1, doneWords1) == False and MainComparison.__wordInList(word2, doneWords2) == False):
                    comparison.append(MainComparison.findComparison(model, word1, word2))
                    doneWords1.append(word1)
                    doneWords2.append(word2)

            comparison = [x for x in comparison if x != None]
            if(len(comparison) == 0):
                return None
            return comparison

        except Exception as ae:
            print(ae)

    def loadModel(name):
        try:
            npzfile = numpy.load(name)
            return npzfile

        except Exception as ae:
            print(ae)

    def saveModel(name, model):
        try:
            ndSentences = model['ndSentences']
            baseWords = model['baseWords']
            numpy.savez(name, ndSentences = ndSentences, baseWords = baseWords)

        except Exception as ae:
            print(ae)

    def trainModel(sentences):
        try:
            #Input as sentences
            ndSentences, baseWords = MainComparison.__selectBaseWord(sentences)

            #Saving arrays in one
            numpy.savez('TrainedModel1.npz', ndSentences = ndSentences, baseWords = baseWords)
        
            model = {}
            with numpy.load('TrainedModel1.npz') as a:
                model['ndSentences'] = a['ndSentences']
                model['baseWords'] = a['baseWords']
            os.remove('TrainedModel1.npz')
            return model

        except Exception as ae:
            print(ae)
            return None

    def retrainModel(sentences, npzfile):
        try:
            ndSentences, baseWords = MainComparison.__selectBaseWord(sentences, npzfile['ndSentences'], npzfile['baseWords'])
            numpy.savez('ReTrainedModel1.npz', ndSentences = ndSentences, baseWords = baseWords)

            model = {}
            with numpy.load('ReTrainedModel1.npz') as a:
                model['ndSentences'] = a['ndSentences']
                model['baseWords'] = a['baseWords']
            os.remove('ReTrainedModel1.npz')
            return model

        except Exception as ae:
            print(ae)
            return None

    def __groupStringToPairs(groupString):
        try:
            pairs = groupString.split(",")
            pairs = [x for x in pairs if x != '']
            return pairs

        except Exception as ae:
            print(ae)

    def __groupStringToWords(groupString):
        try:
            pairs = groupString.split(",")
            words = []
            for i in pairs:
                words.extend(i.split(" "))
            
            words = [x for x in words if x != '']
            return words

        except Exception as ae:
            print(ae)

    def __wordInList(word, list):
        try:
            #Input should be string and list
            if(len(list) == 0):
                return False
            for i in list:
                if(i == word):
                    return True
            return False

        except Exception as ae:
            print(ae)