package gr.auth.csd.IRProject;

import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.ml.feature.*;
import org.apache.spark.ml.linalg.SparseVector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.util.ArrayList;


/**
 * Created by steve on 30/12/2016.
 */
public class IRProjectApp {
    public static void main(String[] args) {
        String inputDirectory = "src/main/resources/data.json";
        String ioSVMDirectory = "src/main/resources/tfIdfData.svm";
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

//        NGram ngram = new NGram().setN(2).setInputCol("woutSWords").setOutputCol("ngrams");
//        wordsData = ngram.transform(wordsData);

        HashingTF hashingTF = new HashingTF().setInputCol("woutSWords").setOutputCol("rawFeatures"); //.setNumFeatures(1000);
        Dataset<Row> rawFeaturizedData = hashingTF.transform(wordsData);

        IDF idf = new IDF().setInputCol("rawFeatures").setOutputCol("features");
        IDFModel idfModel = idf.fit(rawFeaturizedData);
        Dataset<Row> featurizedData = idfModel.transform(rawFeaturizedData);

//        featurizedData.select("label", "features").show();

        featurizedData.write().parquet(outputDirectory);
        generateSVMFile(ioSVMDirectory, data);
    }

    private static void generateSVMFile(String ioSVMDirectory, Dataset<Row> data) {
        Dataset<String> ds = data.select("label", "features").map((MapFunction)(Object r) -> {
            Row r1 = (Row) r;
            String label = Long.toString(r1.getLong(0));
            SparseVector sparseVector = (SparseVector) r1.get(1);
            ArrayList<String> toReturn = new ArrayList<>();
            int[] indices = sparseVector.indices();
            for(int i=0;i<indices.length;++i) {
                ++indices[i];
            }
            double[] values = sparseVector.values();
            for(int i=0;i<indices.length;++i) {
                toReturn.add(String.format("%d:%f", indices[i], values[i]));
            }
            return label + " " + String.join(" ", toReturn);
        }, Encoders.STRING());
        ds.write().text(ioSVMDirectory);
    }
}
