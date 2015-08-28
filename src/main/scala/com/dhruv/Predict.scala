package com.dhruv

import org.apache.spark.SparkConf
import org.apache.spark.mllib.classification.NaiveBayesModel
import org.apache.spark.streaming.twitter._
import org.apache.spark.streaming.{Seconds, StreamingContext}

/**
 * Pulls live tweets and predicts the sentiment.
 */
object Predict {
  def main(args: Array[String]) {
    if (args.length < 1) {
      System.err.println("Usage: " + this.getClass.getSimpleName + " <modelDirectory> ")
      System.exit(1)
    }

    val Array(modelFile) =
      Utils.parseCommandLineWithTwitterCredentials(args)

    println("Initializing Streaming Spark Context...")
    val conf = new SparkConf().setAppName(this.getClass.getSimpleName)
    val ssc = new StreamingContext(conf, Seconds(5))

    println("Initializing Twitter stream...")
    val tweets = TwitterUtils.createStream(ssc, Utils.getAuth)
    val statuses = tweets.filter(_.getLang == "en").map(_.getText)

    println("Initalizaing the Naive Bayes model...")
    val model = NaiveBayesModel.load(ssc.sparkContext, modelFile.toString)

    val labeled_statuses = statuses
      .map(t => (t, model.predict(Utils.featurize(t))))

    labeled_statuses.print()

    // Start the streaming computation
    println("Initialization complete.")
    ssc.start()
    ssc.awaitTermination()
  }
}
