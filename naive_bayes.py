from __future__ import division
from os import listdir
from os.path import isfile, join
from collections import Counter
import sys
import math
import random

def path_to_files(directory_path):
    files = [ f for f in listdir(directory_path) if isfile(join(directory_path,f))]
    return files

def split_files(list_of_files):
    x = random.randint(0, len(list_of_files))
    y = random.randint(x, len(list_of_files))
    a = list_of_files[:x]
    b = list_of_files[x:y]
    c = list_of_files[y:len(list_of_files)]
    return [a, b, c]

def getCrossValidationSplits(iteration, splits):
    trainingFiles = splits[iteration % 3] + splits[(iteration + 1) % 3]
    testingFiles = splits[(iteration + 2) % 3]
    return trainingFiles, testingFiles


def predict(wordList, pwc, p_c):
    unique_words_hist = Counter(wordList)
    sum = 0
    for word in unique_words_hist.keys():
        if word not in pwc:
            P_w_c = pwc["UNK"]
        else:
            P_w_c = pwc[word]
        try:
            lg = math.log(P_w_c)
            sum += unique_words_hist[word] * lg
        except ValueError:
            print "error! P_w_c = ", P_w_c
    return math.log(p_c) + sum


def naive_bayes(rootDir):
    posDir = rootDir + "pos/"
    negDir = rootDir + "neg/"
    posFiles = path_to_files(posDir)
    negFiles = path_to_files(negDir)
    acc = 0
    for i in range(3):
        print "iteration ", i+1, ":"
        posSplits = split_files(posFiles)
        negSplits = split_files(negFiles)
        posTrainingFiles, posTestingFiles = getCrossValidationSplits(i, posSplits)
        negTrainingFiles, negTestingFiles = getCrossValidationSplits(i, negSplits)
        pwc_pos = {}
        count_w_c_pos = Counter()
        count_c_pos = 0
        for file in posTrainingFiles:
            f = open(posDir + file)
            words = f.read()
            count_w_c_pos += Counter(words)
            count_c_pos += len(words)
        v_pos = len(count_w_c_pos.keys())
        pwc_neg = {}
        count_w_c_neg = Counter()
        count_c_neg = 0
        for file in negTrainingFiles:
            f = open(negDir + file)
            words = f.read()
            count_w_c_neg += Counter(words)
            count_c_neg += len(words)
        v_neg = len(count_w_c_neg.keys())
        for word in count_w_c_pos.keys():
            pwc_pos[word] = (count_w_c_pos[word] + 1) / (count_c_pos + v_pos + 1)
        pwc_pos["UNK"] = 1 / (count_c_pos + v_pos + 1)
        for word in count_w_c_neg.keys():
            pwc_neg[word] = (count_w_c_neg[word] + 1) / (count_c_neg + v_neg + 1)
        pwc_neg["UNK"] = 1 / (count_c_neg + v_neg + 1)
        pos_correct = 0
        neg_correct = 0
        p_c_pos = len(posTrainingFiles) / (len(posTrainingFiles) + len(negTrainingFiles))
        p_c_neg = len(negTrainingFiles) / (len(posTrainingFiles) + len(negTrainingFiles))
        for file in posTestingFiles:
            f = open(posDir + file)
            words = f.read()
            positive_pred_val = predict(words, pwc_pos, p_c_pos)
            negative_pred_val = predict(words, pwc_neg, p_c_neg)
            if positive_pred_val > negative_pred_val:
               pos_correct += 1
        print "num_pos_test_docs:", len(posTestingFiles)
        print "num_pos_training_docs:", len(posTrainingFiles)
        print "num_pos_correct_docs:", pos_correct
        for file in negTestingFiles:
            f = open(negDir + file)
            words = f.read()
            positive_pred_val = predict(words, pwc_pos, p_c_pos)
            negative_pred_val = predict(words, pwc_neg, p_c_neg)
            if negative_pred_val > positive_pred_val:
               neg_correct += 1
        print "num_neg_test_docs:", len(negTestingFiles)
        print "num_neg_training_docs:", len(negTrainingFiles)
        print "num_neg_correct_docs:", neg_correct
        total_test_docs = len(negTestingFiles) + len(posTestingFiles)
        accuracy = 100 * ((pos_correct + neg_correct) / total_test_docs)
        acc += accuracy
        print "accuracy: ", accuracy, "%"
    print "ave_accuracy:", acc/3, "%"


def main():
    if len(sys.argv) != 2:
        print "usage is: python naive_bayes.py <full path to review directory>"
        sys.exit(1)
    naive_bayes(sys.argv[1])


if __name__ == "__main__":
    main()
