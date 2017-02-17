package gr.auth.csd.IRProject.unsupervised;

import org.apache.log4j.Level;
import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;
import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

/**
 * Uses the KMeans Algorithm to predict the class (positive, negative) of the movie reviews.
 * After identifying the data clusters, we use majority vote to determine which class it belongs to.
 */
public class KMeansAlgorithm {
    static String inputDirectory = "src/main/resources/w2vData";

    static Logger logger;
    static {
        logger = LogManager.getRootLogger();
        logger.setLevel(Level.WARN);
    }

    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder()
                                         .appName("IRProjectKMeans")
                                         .getOrCreate();

        Dataset<Row> data = spark.read().parquet(inputDirectory);

        // 90-10% hold out validation
        Dataset<Row>[] datasets = data.randomSplit(new double[]{0.9,0.1}, 123321);
        Dataset<Row> trainSet = datasets[0];
        Dataset<Row> testSet = datasets[1];

        KMeans kmeans = new KMeans().setFeaturesCol("w2vRes").setK(2).setSeed(2L);
        KMeansModel model = kmeans.fit(trainSet);

        // Use spark temp SQL table in order to calculate accuracy.
        Dataset<Row> predictionsTest = model.transform(testSet);
        predictionsTest.registerTempTable("pred_test");
        Dataset<Row> predictionsTrain = model.transform(trainSet);
        predictionsTrain.registerTempTable("pred_train");

        double sameTest = spark.sql("select * from pred_test where label=prediction").count();
        double sameTrain = spark.sql("select * from pred_train where label=prediction").count();
        double samePrctTrain = sameTrain/predictionsTrain.count();
        double samePrctTest = sameTest/predictionsTest.count();

        double accuracyOnTestSet;
        double accuracyOnTrainSet;

        // Use majority vote to assign clusters to classes
        if (samePrctTrain < 0.5) {
            accuracyOnTrainSet = 1 - samePrctTrain;
            accuracyOnTestSet = 1 - samePrctTest;
        }
        else {
            accuracyOnTrainSet = samePrctTrain;
            accuracyOnTestSet = samePrctTest;
        }

        // Print accuracy metrics, WSSE and cluster centers in DEBUG mode.
        logger.log(Level.WARN, String.format("Accuracy(Train): %f", accuracyOnTrainSet));
        logger.log(Level.WARN, String.format("Accuracy(Test): %f", accuracyOnTestSet));

        double WSSSE = model.computeCost(data);
        logger.log(Level.WARN, "Within Set Sum of Squared Errors = " + WSSSE);

        Vector[] centers = model.clusterCenters();
        logger.log(Level.DEBUG, "Cluster Centers: ");
        for (Vector center: centers) {
            logger.log(Level.DEBUG, center);
        }

        spark.stop();
    }
}
