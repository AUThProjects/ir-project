package gr.auth.csd.IRProject;

import org.apache.spark.ml.feature.*;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;


/**
 * Created by steve on 30/12/2016.
 */
public class IRProjectApp {
    public static void main(String[] args) {
        String inputDirectory = "src/main/resources/data.json";
        String outputDirectory = "src/main/resources/tfIdfData.json";
        SparkSession spark = SparkSession.builder()
                                         .appName("IRProject")
                                         .getOrCreate();

        Dataset<Row> data = spark.read().json(inputDirectory);

        String regex = "(\\<.+\\>|\\W|[.,?:;!()])+";
        RegexTokenizer tokenizer = new RegexTokenizer().setInputCol("review").setOutputCol("words").setPattern(regex);
        Dataset<Row> wordsData = tokenizer.transform(data);

        StopWordsRemover swr = new StopWordsRemover().setInputCol("words").setOutputCol("woutSWords");
        wordsData = swr.transform(wordsData);

        NGram ngram = new NGram().setN(3).setInputCol("woutSWords").setOutputCol("ngrams");
        wordsData = ngram.transform(wordsData);

        HashingTF hashingTF = new HashingTF().setInputCol("ngrams").setOutputCol("rawFeatures"); //.setNumFeatures(1000);
        Dataset<Row> rawFeaturizedData = hashingTF.transform(wordsData);

        IDF idf = new IDF().setInputCol("rawFeatures").setOutputCol("features");
        IDFModel idfModel = idf.fit(rawFeaturizedData);
        Dataset<Row> featurizedData = idfModel.transform(rawFeaturizedData);

//        featurizedData.select("label", "features").show();

        featurizedData.write().parquet(outputDirectory);
    }
}
