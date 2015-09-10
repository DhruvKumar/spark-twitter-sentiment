package com.dhruv

import org.apache.spark.deploy.SparkSubmit

/**
 * Wrapper around Spark Submit for fast development and debugging in IntelliJ (and other IDEs).
 * If this class is used as the main class in a run configuration, then
 * you can pass in arguments exactly like you would to the "spark-submit"
 * script and launch the job from IntelliJ! To use in IntelliJ IDEA:
 * <ol>
 *   <li>
 *     Add $SPARK_HOME/conf/ folder as a resource folder to your module
 *   </li>
 *   <li>
 *     Add the following jars in "runtime" scope to the module's dependency list:
 *        - $SPARK_HOME/lib/spark-assembly-xxxx.jar
 *        - $SPARK_HOME/lib/datanucleus*.jar (all 3 datanucleus jars)
 *   </li>
 *   <li>
 *     Create a new run config, set SparkSubmitWrapper as the main class
 *     and pass in the arguments using Program Arguments box. Eg:
 *       --class
 *       com.dhruv.Train
 *       --master
 *       local[*]
 *       --driver-memory
 *       1g
 *       --executor-memory
 *       4g
 *       target/twittersentiment-0.0.1-jar-with-dependencies.jar
 *       /Users/dkumar/code/sentiment-analysis/datasets/sentiment-analysis-dataset/small.csv
 *   </li>
 *   <li>
 *     Optional: in the Run Config, add any extra actions before launching, eg: mvn clean package.
 *     This will cause the action to be executed each time you run or debug, before anything else.
 *     Maven packaging is not needed in local testing, just the default "make" action is sufficient.
 *   </li>
 *   <li>
 *     Hit run to execute. You can also add breakpoints and do a debug to introspect object values.
 *   </li>
 */
object SparkSubmitWrapper {
  def main(args: Array[String]): Unit = {
    SparkSubmit.main(args)
  }
}
