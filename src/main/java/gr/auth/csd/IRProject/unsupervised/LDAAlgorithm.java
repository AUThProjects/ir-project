package gr.auth.csd.IRProject.unsupervised;

import java.io.IOException;

import org.apache.log4j.Level;
import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;
import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.ml.clustering.LDA;
import org.apache.spark.ml.clustering.LDAModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

/**
 * Uses the Latent Dirichlet Allocation Algorithm to predict the class (positive, negative) of the movie reviews.
 * After identifying the data clusters, we use majority vote to determine which class it belongs to.
 */
public class LDAAlgorithm {
    static String inputDirectory = "src/main/resources/w2vData";
    static String outputDirectory = "src/main/resources/LDAModel";

    static Logger logger;
    static {
        logger = LogManager.getRootLogger();
        logger.setLevel(Level.WARN);
    }

    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder()
                .appName("IRProjectLDA")
                .getOrCreate();

        Dataset<Row> data = spark.read().parquet(inputDirectory);

        // 90-10 hold-out validation
        Dataset<Row>[] datasets = data.randomSplit(new double[]{0.9,0.1}, 123321);
        Dataset<Row> trainSet = datasets[0];
        Dataset<Row> testSet = datasets[1];

        LDA lda = new LDA().setFeaturesCol("w2vRes").setK(2).setMaxIter(50); //.setOptimizer("online");
        LDAModel model = lda.fit(trainSet);
        try {
            model.save(outputDirectory);
        }
        catch (IOException e) {
            logger.log(Level.ERROR, e.getMessage());
        }

        // Display log likelihood and perplexity metrics, in DEBUG mode.
        double ll = model.logLikelihood(trainSet);
        double lp = model.logPerplexity(trainSet);
        logger.log(Level.DEBUG, "The lower bound on the log likelihood of the entire corpus: " + ll);
        logger.log(Level.DEBUG, "The upper bound bound on perplexity: " + lp);

        // Saving LDA model.
        try {
            model.save(outputDirectory);
        }
        catch (IOException e) {
            logger.log(Level.ERROR, e.getMessage());
        }

        Dataset<Row> predictionsTest = model.transform(testSet);
        Dataset<Row> predictionsTrain = model.transform(trainSet);

        Dataset<Integer> resultsTrain =  predictionsTrain.select("topicDistribution", "label").map(
                new MapFunction<Row, Integer>() {
                    public Integer call(Row r) {
                        double[] l = ((org.apache.spark.ml.linalg.DenseVector)r.get(0)).toArray();
                        Integer res = (l[0] > l[1]) ? 0 : 1;
                        return (res == r.getLong(1)) ? 1 : 0;
                    }}, Encoders.INT());

        Dataset<Integer> resultsTest =  predictionsTest.select("topicDistribution", "label").map(
                new MapFunction<Row, Integer>() {
                    public Integer call(Row r) {
                        double[] l = ((org.apache.spark.ml.linalg.DenseVector)r.get(0)).toArray();
                        Integer res = (l[0] > l[1]) ? 0 : 1;
                        return (res == r.getLong(1)) ? 1 : 0;
                    }}, Encoders.INT());

        double sameTest = resultsTest.where("value=1").count();
        double sameTrain = resultsTrain.where("value=1").count();
        double samePrctTrain = sameTrain/predictionsTrain.count();
        double samePrctTest = sameTest/predictionsTest.count();

        double accuracyOnTestSet;
        double accuracyOnTrainSet;

        // Using majority vote to assign clusters to classes.
        if (samePrctTrain < 0.5) {
            accuracyOnTrainSet = 1 - samePrctTrain;
            accuracyOnTestSet = 1 - samePrctTest;
        }
        else {
            accuracyOnTrainSet = samePrctTrain;
            accuracyOnTestSet = samePrctTest;
        }

        logger.log(Level.WARN, String.format("Accuracy(Train): %f", accuracyOnTrainSet));
        logger.log(Level.WARN, String.format("Accuracy(Test): %f", accuracyOnTestSet));

        spark.stop();
    }
}
