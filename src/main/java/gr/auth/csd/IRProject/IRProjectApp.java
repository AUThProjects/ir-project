package gr.auth.csd.IRProject;

import java.util.ArrayList;

import org.apache.log4j.Level;
import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;
import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.ml.feature.*;
import org.apache.spark.ml.linalg.SparseVector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

/**
 * TF-IDF and SVM formatting transformation class.
 */
public class IRProjectApp {

    static Logger logger;

    static {
        logger = LogManager.getRootLogger();
        logger.setLevel(Level.WARN);
    }

    /**
     * Execution entrypoint.
     * @param args CLI Arguments
     */
    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder()
                                         .appName("IRProject")
                                         .getOrCreate();

        trainSetDataGeneration(spark);
        testSetDataGeneration(spark);
    }

    /**
     * Generates the train set data from the input json file.
     * @param spark Spark context object
     */
    private static void trainSetDataGeneration(SparkSession spark) {
        String inputDirectory = "src/main/resources/data.json";
        String ioSVMDirectory = "src/main/resources/tfIdfData.svm";
        String outputDirectory = "src/main/resources/tfIdfData.json";

        Dataset<Row> data = spark.read().json(inputDirectory);
        Dataset<Row> featurizedData = generateTfIdfFile(outputDirectory, data);
        generateSVMFile(ioSVMDirectory, featurizedData);
    }

    /**
     * Generates the test set data from the input json file.
     * @param spark Spark context object
     */
    private static void testSetDataGeneration(SparkSession spark) {
        String inputDirectory = "src/main/resources/data_unlabelled.json";
        String ioSVMDirectory = "src/main/resources/tfIdfDataUnlabelled.svm";
        String outputDirectory = "src/main/resources/tfIdfDataUnlabelled.json";

        Dataset<Row> data = spark.read().json(inputDirectory);
        Dataset<Row> featurizedData = generateTfIdfFile(outputDirectory, data, 2);
        generateSVMFile(ioSVMDirectory, featurizedData);
    }

    /**
     * Generates the TF-IDF weights from the input dataset.
     * @param outputDirectory The output directory of the vector model analysis
     * @param data The input data as a Dataset
     * @param ngrams How many words to take in ngrams
     * @return An augmented dataset containing without stop words, containing ngrams and td, idf weights.
     */
    private static Dataset<Row> generateTfIdfFile(String outputDirectory, Dataset<Row> data, int ngrams) {
        // Regex for recognizing words as tokens
        String regex = "(\\<.+\\>|\\W|[.,?:;!()])+";
        RegexTokenizer tokenizer = new RegexTokenizer().setInputCol("review").setOutputCol("words").setPattern(regex);
        Dataset<Row> wordsData = tokenizer.transform(data);

        // Remove stop words
        String[] stopWords = new String[]{"i","me","my","myself","we","our","ours","ourselves","you","your","yours","yourself","yourselves","he","him","his","himself","she","her","hers","herself","it","its","itself","they","them","their","theirs","themselves","what","which","who","whom","this","that","these","those","am","is","are","was","were","be","been","being","have","has","had","having","do","does","did","doing","a","an","the","and","but","if","or","because","as","until","while","of","at","by","for","with","about","against","between","into","through","during","before","after","above","below","to","from","up","down","in","out","on","off","over","under","again","further","then","once","here","there","when","where","why","how","all","any","both","each","few","more","most","other","some","such","only","own","same","so","than","too","very","s","t","can","will","just","don","should","now","d","ll","m","o","re","ve","y"};
        StopWordsRemover swr = new StopWordsRemover().setStopWords(stopWords).setInputCol("words").setOutputCol("woutSWords");
        wordsData = swr.transform(wordsData);

        // Take 2-grams
        NGram ngram = new NGram().setN(ngrams).setInputCol("woutSWords").setOutputCol("ngrams");
        wordsData = ngram.transform(wordsData);

        // Calculate TFs
        HashingTF hashingTF = new HashingTF().setInputCol("woutSWords").setOutputCol("rawFeatures"); //.setNumFeatures(1000);
        Dataset<Row> rawFeaturizedData = hashingTF.transform(wordsData);

        // Calculate IDFs
        IDF idf = new IDF().setInputCol("rawFeatures").setOutputCol("features");
        IDFModel idfModel = idf.fit(rawFeaturizedData);
        Dataset<Row> featurizedData = idfModel.transform(rawFeaturizedData);

        // Write output as parquet formatted file
        featurizedData.write().parquet(outputDirectory);
        return featurizedData;
    }

    /**
     * Generates an SVM format file from the input dataset. This format is needed for some algorithms.
     * @param ioSVMDirectory The output directory of the SVM formatted file.
     * @param data The input data as a Dataset
     */
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
