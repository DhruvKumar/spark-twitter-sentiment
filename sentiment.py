from pyspark.mllib.classification import NaiveBayes
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.feature import IDF

def parseTweet(line):
    parts = line.split(',')
    label = float(parts[1])
    tweet = parts[3]
    words = tweet.strip().split(" ")
    return (label, words)

allData = sc.textFile("/Users/dkumar/code/sentiment-analysis/datasets/sentiment-analysis-dataset/dataset.csv")

header = allData.first()

data = allData.filter(lambda x: x != header).map(parseTweet)

training, test = data.randomSplit([0.7, 0.3], seed=0)

hashingTF = HashingTF()

tf_training = training.map(lambda tup: hashingTF.transform(tup[1]))

idf_training = IDF().fit(tf_training)

tfidf_training = idf_training.transform(tf_training)

tfidf_idx = tfidf_training.zipWithIndex()

training_idx = training.zipWithIndex()

idx_training = training_idx.map(lambda line: (line[1], line[0]))

idx_tfidf = tfidf_idx.map(lambda l: (l[1], l[0]))

joined_tfidf_training = idx_training.join(idx_tfidf)

training_labeled = joined_tfidf_training.map(lambda tup: tup[1])

labeled_training_data = training_labeled.map(lambda k: LabeledPoint(k[0][0], k[1]))

model = NaiveBayes.train(labeled_training_data, 1.0)

tf_test = test.map(lambda tup: hashingTF.transform(tup[1]))

idf_test = IDF().fit(tf_test)

tfidf_test = idf_test.transform(tf_test)

tfidf_idx = tfidf_test.zipWithIndex()

test_idx = test.zipWithIndex()

idx_test = test_idx.map(lambda line: (line[1], line[0]))

idx_tfidf = tfidf_idx.map(lambda l: (l[1], l[0]))

joined_tfidf_test = idx_test.join(idx_tfidf)

test_labeled = joined_tfidf_test.map(lambda tup: tup[1])

labeled_test_data = test_labeled.map(lambda k: LabeledPoint(k[0][0], k[1]))

predictionAndLabel = labeled_test_data.map(lambda p : (model.predict(p.features), p.label))

accuracy = 1.0 * predictionAndLabel.filter(lambda (x, v): x == v).count() / labeled_test_data.count()

print accuracy

