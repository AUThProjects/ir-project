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

        String[] stopWords = new String[]{"i","me","my","myself","we","our","ours","ourselves","you","your","yours","yourself","yourselves","he","him","his","himself","she","her","hers","herself","it","its","itself","they","them","their","theirs","themselves","what","which","who","whom","this","that","these","those","am","is","are","was","were","be","been","being","have","has","had","having","do","does","did","doing","a","an","the","and","but","if","or","because","as","until","while","of","at","by","for","with","about","against","between","into","through","during","before","after","above","below","to","from","up","down","in","out","on","off","over","under","again","further","then","once","here","there","when","where","why","how","all","any","both","each","few","more","most","other","some","such","only","own","same","so","than","too","very","s","t","can","will","just","don","should","now","d","ll","m","o","re","ve","y"};
        StopWordsRemover swr = new StopWordsRemover().setStopWords(stopWords).setInputCol("words").setOutputCol("woutSWords");
        wordsData = swr.transform(wordsData);

        NGram ngram = new NGram().setN(2).setInputCol("woutSWords").setOutputCol("ngrams");
        wordsData = ngram.transform(wordsData);

        HashingTF hashingTF = new HashingTF().setInputCol("ngrams").setOutputCol("rawFeatures"); //.setNumFeatures(1000);
        Dataset<Row> rawFeaturizedData = hashingTF.transform(wordsData);

        IDF idf = new IDF().setInputCol("rawFeatures").setOutputCol("features");
        IDFModel idfModel = idf.fit(rawFeaturizedData);
        Dataset<Row> featurizedData = idfModel.transform(rawFeaturizedData);

//        featurizedData.select("label", "features").show();

        featurizedData.write().parquet(outputDirectory);
        generateSVMFile(ioSVMDirectory, featurizedData);
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
        ds.coalesce(1).write().text(ioSVMDirectory);
    }
}
