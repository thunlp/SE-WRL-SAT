# coding:utf8

'''
Calculate Chinese Word Similarity Scores with Wordsim-240 and Wordsim-297 Datasets
'''

import sys
import numpy as np
from numpy import linalg

wordvecFile = sys.argv[1]

wordVecDict = {}
file = open(wordvecFile)
tmp = file.readline()
num = 0
for line in file:
    num += 1
    if num % 500 == 0:
        print("Reading the %d-th word" % num)

    items = line.strip().split()
    word = items[0]
    vec = list(map(float, items[1:]))
    if linalg.norm(vec) != 0:
        wordVecDict[word] = vec / linalg.norm(vec)
file.close()
print('Word embeddings reading completes and total number of words is:', num)


wordSimType = ['240', '297']
for x in wordSimType:
    file = open('data/wordsim-' + x + '.txt')
    testPairNum = 0
    skipPairNum = 0

    wordSimStd = []
    wordSimPre = []
    for line in file:
        word1, word2, valStr = line.strip().split()
        if (word1 in wordVecDict) and (word2 in wordVecDict):
            testPairNum += 1
            wordSimStd.append(float(valStr))
            wordVec1 = wordVecDict[word1]
            wordVec2 = wordVecDict[word2]
            cosSim = np.dot(wordVec1, wordVec2)
            wordSimPre.append(cosSim)
        else:
            skipPairNum += 1
            # print('Skip:', word1, word2)
    corrCoef = np.corrcoef(wordSimStd, wordSimPre)[0, 1]
    print("WordSim-" + x + " Score:", corrCoef)
    print('TestPair:', testPairNum, 'SkipPair:', skipPairNum)
    file.close()
