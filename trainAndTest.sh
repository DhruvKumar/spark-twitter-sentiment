spark-submit --class com.dhruv.Train --master local[*] --driver-memory 1g --executor-memory 4g target/twittersentiment-0.0.1.jar /Users/dkumar/code/sentiment-analysis/datasets/sentiment-analysis-dataset/dataset.csv
  
spark-submit \
     --class "com.dhruv.Predict" \
     --master local[*] \
     target/twittersentiment-0.0.1-jar-with-dependencies.jar \
     /Users/dkumar/code/sentiment-analysis/datasets/models \
     --consumerKey 1OYIOzdy9B9pucIfz3am68jc \
     --consumerSecret 6Fwm7UFzOweq8PwYHKhIKmMPzfMhYcWyVbAKoqEob7e3HXOQT \
     --accessToken 336943510-gAo2UbccZtaApq7UIlwndC0warzNjc1J3LWxgav  \
     --accessTokenSecret QoFWIJuLerryTFTO1int1wXqhlM869FqsZJpU9F68s5e
