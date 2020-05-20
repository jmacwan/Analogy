import Analogy.ComparisonPOSNumpy

def findComparison(model, word1, word2):
    try:
        return ComparisonPOSNumpy.MainComparison.findComparison(model, word1, word2)
    except Exception as ae:
        print(ae)

def findSentenceComparison(model, sentence1, sentence2):
    try:
        return ComparisonPOSNumpy.MainComparison.findSentenceComparison(model, sentence1, sentence2)
    except Exception as ae:
        print(ae)

def loadModel(name):
    try:
        return ComparisonPOSNumpy.MainComparison.loadModel(name)
    except Exception as ae:
        print(ae)

def saveModel(name, model):
    try:
        return ComparisonPOSNumpy.MainComparison.saveModel(name, model)
    except Exception as ae:
        print(ae)

def trainModel(sentences):
    try:
        return ComparisonPOSNumpy.MainComparison.trainModel(sentences)
    except Exception as ae:
        print(ae)

def retrainModel(sentences, npzfile):
    try:
        return ComparisonPOSNumpy.MainComparison.retrainModel(sentences, npzfile)
    except Exception as ae:
        print(ae)