package com.dhruv

import java.io.InputStream
import java.util.Properties

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.classification.NaiveBayes

import edu.stanford.nlp.ling.CoreAnnotations.{LemmaAnnotation, SentencesAnnotation, TokensAnnotation}
import edu.stanford.nlp.pipeline.{Annotation, StanfordCoreNLP}

import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer

object Train {
  def main(args: Array[String]) {
    if (args.length == 0) {
      System.err.println("Usage: " + this.getClass.getSimpleName + " <training file> ")
      System.exit(1)
    }

    val sparkConf = new SparkConf().
      setAppName("Twitter Sentiment Analyzer")
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    val sc = new SparkContext(sparkConf)
    val stopWords = sc.broadcast(loadStopWords("/stopwords.txt")).value
    val allData = sc.textFile(args(0))
    val header = allData.first()
    val data = allData.filter(x => x != header)
    val splits = data.randomSplit(Array(0.8, 0.2), seed = 11L)
    val training = splits(0)
    val test = splits(1)

    def toLabels(line: String) = {
      val words = line.split(',')
      (words(1), words(3))
    }

    val hashingTF = new HashingTF(1000)

    val training_labeled = training.map(x => toLabels(x)).
      map(t => (t._1, Utils.featurize(t._2))).
      map(x => new LabeledPoint((x._1).toDouble, x._2))

    def time[R](block: => R): R = {
      val t0 = System.nanoTime()
      val result = block    // call-by-name
      val t1 = System.nanoTime()
      println("\n\nElapsed time: " + (t1 - t0)/1000 + "ms")
      result
    }

    println("\n\n********* Training **********\n\n")
    val model = time { NaiveBayes.train(training_labeled, 1.0) }

    println("\n\n********* Testing **********\n\n")
    val testing_labeled = test.map(x => toLabels(x)).
      map(t => (t._1, Utils.featurize(t._2), t._2)).
      map(x => {
      val lp = new LabeledPoint((x._1).toDouble, x._2)
      (lp, x._3)
    })

    val predictionAndLabel = time { testing_labeled.map(p => {
      val labeledPoint = p._1
      val text = p._2
      val features = labeledPoint.features
      val actual_label = labeledPoint.label
      val predicted_label = model.predict(features)
      (actual_label, predicted_label, text)
    }) }

    val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / test.count()

    println("Training and Testing complete. Accuracy is = " + accuracy)
    println("\nSome Predictions:\n")

    predictionAndLabel.take(10).foreach( x => {
      println("---------------------------------------------------------------")
      println("Text = " + x._3)
      println("Actual Label = " + (if (x._1 == 1) "positive" else "negative"))
      println("Predicted Label = " + (if (x._2 == 1) "positive" else "negative"))
      println("----------------------------------------------------------------\n\n")
    } )

    if (args.length == 2) {
      val savePath = args(1) + "/" + accuracy.toString
      model.save(sc, args(1))
      println("\n\n********* Model saved to: " + savePath + "\n\n")
    }
    sc.stop()
    println("\n\n********* Stopped Spark Context succesfully, exiting ********")
  }


  /**
   * Methods included for future extension. Some ideas:
   * - use stopwords.txt to weed out "the", "in", etc.
   * - lemmify text
   */

  def createNLPPipeline(): StanfordCoreNLP = {
    val props = new Properties()
    props.put("annotators", "tokenize, ssplit, pos, lemma")
    new StanfordCoreNLP(props)
  }

  def plainTextToLemmas(text: String, stopWords: Set[String], pipeline: StanfordCoreNLP)
  : Seq[String] = {
    val doc = new Annotation(text)
    pipeline.annotate(doc)
    val lemmas = new ArrayBuffer[String]()
    val sentences = doc.get(classOf[SentencesAnnotation])
    for (sentence <- sentences.asScala;
         token <- sentence.get(classOf[TokensAnnotation]).asScala) {
      val lemma = token.get(classOf[LemmaAnnotation])
      if (lemma.length > 2 && !stopWords.contains(lemma) && isOnlyLetters(lemma)) {
        lemmas += lemma.toLowerCase
      }
    }
    lemmas
  }

  def isOnlyLetters(str: String): Boolean = {
    var i = 0
    while (i < str.length) {
      if (!Character.isLetter(str.charAt(i))) {
        return false
      }
      i += 1
    }
    true
  }

  def loadStopWords(path: String) = {
    val stream: InputStream = getClass.getResourceAsStream(path)
    val lines = scala.io.Source.fromInputStream(stream).getLines
    lines.toSet
  }

}
