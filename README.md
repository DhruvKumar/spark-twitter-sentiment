Sentiment Analysis of Live Twitter Stream Using Apache Spark 
============================================================

This application analyzes live tweets and predicts if they are positive, or negative. The application works by connecting 
to the Twitter stream, and applying a model built offline using Spark's machine learning library (Mllib) to classify 
the tweet's sentiment. Using the instructions on this page, you will be able to build the model on HDP Sandbox and then 
apply it to a live twitter stream.

Prerequisites
-------------

* Download the HDP 2.3 Sandbox from [here](http://hortonworks.com/products/hortonworks-sandbox/#install)
* Start the Sandbox, and add its IP address into your local machine's /etc/hosts file:

```bash
$ sudo echo "172.16.139.139 sandbox.hortonworks.com" >> /etc/hosts
```

* Log into the sandbox, and clone this repository:  

```bash
$ ssh root@sandbox.hortonworks.com
$ cd
$ git clone https://github.com/DhruvKumar/spark-twitter-sentiment
```

* Download the labeled training data into the sandbox. 

```bash
$ wget https://www.dropbox.com/s/1k355mod4p70jiq/dataset.csv?dl=0
```

* Put the tweet data into hdfs, at a location /tmp/tweets

```bash
$ hadoop fs -put tweets /tmp/tweets
```

* Sign up for dev Twitter account and get the OAuth credentials [here for free](https://apps.twitter.com/). 


Build and package the code
-----------------------------------------

Compile the code using maven:

```bash
$ cd
$ cd spark-twitter-sentiment
$ mvn clean package 
``` 

This will build and place the uber jar "twittersentiment-0.0.1-jar-with-dependencies.jar" under the target/ directory.
 
We're now ready to train the model

Train the Model 
-----------------------------------------

```bash
$ spark-submit --master yarn-client \
               --driver-memory 1g \
               --executor-memory 2g \
               target/twittersentiment-0.0.1-jar-with-dependencies.jar \
               hdfs://tmp/tweets/dataset.csv
               trainedModel
```

This will train and test the model, and put it under the trainedModel directory. You should see the results of the 
testing, with predicted sentiments like this:

```bash

********* Training **********


Elapsed time: 13868063ms


********* Testing **********


Elapsed time: 16326ms
Training and Testing complete. Accuracy is = 0.6536062932423466

Some Predictions:

---------------------------------------------------------------
Text =          i think mi bf is cheating on me!!!       T_T
Actual Label = negative
Predicted Label = negative



---------------------------------------------------------------
Text =       handed in my uniform today . i miss you already
Actual Label = positive
Predicted Label = negative


---------------------------------------------------------------
Text =       I must think about positive..
Actual Label = negative
Predicted Label = negative



---------------------------------------------------------------
Text =       thanks to all the haters up in my face all day! 112-102
Actual Label = positive
Predicted Label = positive



---------------------------------------------------------------
Text =     &lt;-------- This is the way i feel right now...
Actual Label = negative
Predicted Label = positive



---------------------------------------------------------------
Text =     HUGE roll of thunder just now...SO scary!!!!
Actual Label = negative
Predicted Label = positive



---------------------------------------------------------------
Text =     You're the only one who can see this cause no one else is following me this is for you because you're pretty awesome
Actual Label = positive
Predicted Label = positive



---------------------------------------------------------------
Text =    BoRinG   ): whats wrong with him??     Please tell me........   :-/
Actual Label = negative
Predicted Label = negative



---------------------------------------------------------------
Text =    I didn't realize it was THAT deep. Geez give a girl a warning atleast!
Actual Label = negative
Predicted Label = negative



---------------------------------------------------------------
Text =    i miss you guys too     i think i'm wearing skinny jeans a cute sweater and heels   not really sure   what are you doing today
Actual Label = negative
Predicted Label = negative


********* Stopped Spark Context succesfully, exiting ********

```

Predict sentiment of live tweets using the model
-------------------------------------------------

Now that the model is trained and saved, let's apply it to a live twitter stream and see if we can classify sentiment 
accurately. Launch the following command with your twitter dev keys:

```bash
$ cd 
$ spark-submit \
    --class com.dhruv.Predict \
    --master yarn-client \
    --num-executors 2 \
    --executor-memory 512m \
    --executor-cores 2 \
    target/twittersentiment-0.0.1-jar-with-dependencies.jar \
    trainedModel \ 
    --consumerKey {your Twitter consumer key} \
    --consumerSecret {your Twitter consumer secret} \
    --accessToken {your Twitter access token} \
    --accessTokenSecret {your Twitter access token secret}
```

This command will set up spark streaming, connect to twitter using your dev credentials, and start printing tweets 
with predicted sentiment. Label 1.0 is a positive sentiment, and 0.0 is negative. Each tweet and its predicted label
is displayed like this:

```bash
(#Listened Chasing Pavements by Adele on #MixRadio #ListenNow #Pop #NowPlaying #19 http://t.co/qLXGoq8B8u,1.0)
(Work isn't going so bad, but if I did fix it wtf,0.0)
(RT @RandyAlley: Come on let's have a win Rovers! Good luck lads @Shaun_Lunt @TREVORBC83 http://t.co/tsDEZPrJIO,1.0)
(RT @ribbonchariots: dress shirt ftw!!!! &  the v gets deeper and deeper....... http://t.co/qAL3zIteVF,1.0)
```


Where to go from here?
-------------------------------------------------

I've used a very simple feature extractor--a bigram model, hashed down to 1000 features. This can be vastly improved. 
Experiment with removing stop words (in, the, and, etc.) from the tweets before training as they don't add any info. 
Consider lemmafying the tweets, which makes multiple forms of the word appear as one word (train and trains are same).
I've put in the NLP pipeline parsers and lemma-fiers from Stanford NLP library, so you can start from there. 
Consider also using tf-idf, and experiment with other classifiers in Spark MLlib such as Random Forest.
