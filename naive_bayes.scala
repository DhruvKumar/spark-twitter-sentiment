def parseTweet(line: String) = {
     | val parts = line.split(',')
     | val label = parts(1).toDouble
     | val tweet = parts(3)
     | val words = tweet.trim().split(" ")
     | (label, words)
     | }

val allData = sc.textFile("/Users/dkumar/code/sentiment-analysis/datasets/sentiment-analysis-dataset/small.csv")

val header = allData.first()

val data = allData.filter(x => x != header)

val splits = data.randomSplit(Array(0.7, 0.3), seed = 11L)

val training = splits(0)

val test = splits(1)

def toLabels(line: String) = {
     | val words = line.split(',')
     | (words(1), words(3))
     | }

val training_labeled =  training.map(x => toLabels(x))

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.feature.IDF
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}

val hashingTF = new HashingTF()

val tf_training = training.map(t => hashingTF.transform(t._2))

val idf_training = new IDF().fit(tf_training)

val tfidf_training = idf_training.transform(tf_training)

val tfidf_idx = tfidf_training.zipWithIndex()

val training_idx = training.zipWithIndex()

val idx_training = training_idx.map(t => (t._2, t._1))

val idx_tfidf = tfidf_idx.map(t => (t._2, t._1))

val joined_tfidf_training = idx_training.join(idx_tfidf)

val training_labeled = joined_tfidf_training.map(t => t._2)

val labeled_training_data = training_labeled.map(t => { 
val s = t._1 
val w = s.split(',') 
new LabeledPoint(w(1).toDouble, t._2) 
})

val model = NaiveBayes.train(labeled_training_data, 1.0)

val testing_labeled =  test.map(x => toLabels(x))

val tf_testing = testing_labeled.map(t => hashingTF.transform(t._2))

val idf_test = new IDF().fit(tf_testing)

val tfidf_testing = idf_test.transform(tf_testing)

val tfidf_idx_test = tfidf_testing.zipWithIndex()

val testing_idx = test.zipWithIndex()

val idx_testing = testing_idx.map(t => (t._2, t._1))

val idx_tfidf = tfidf_idx_test.map(t => (t._2, t._1))

val joined_tfidf_testing = idx_testing.join(idx_tfidf)

val testing_labeled = joined_tfidf_testing.map(t => t._2)

val labeled_testing_data = testing_labeled.map(t => { 
val s = t._1 
val w = s.split(',') 
new LabeledPoint(w(1).toDouble, t._2) 
})


val predictionAndLabel = labeled_testing_data.map(p => (model.predict(p.features), p.label))

val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / test.count()

model.save(sc, "/Users/dkumar/code/ml-models/naivebayes")



