package gr.auth.csd.IRProject.unsupervised;

import org.apache.log4j.Level;
import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;
import org.apache.spark.ml.clustering.*;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.io.IOException;

/**
 * Created by steve on 09/01/2017.
 */
public class GaussianMixtureAlgorithm {
    public static void main(String[] args) {
        String inputDirectory = "src/main/resources/w2vData";
        String outputDirectory = "src/main/resources/GaussianMixtureModel";

        SparkSession spark = SparkSession.builder()
                .appName("IRProjectKMeans")
                .getOrCreate();
        Logger logger = LogManager.getRootLogger();
        logger.setLevel(Level.WARN);


        Dataset<Row> data = spark.read().parquet(inputDirectory);

        Dataset<Row>[] datasets = data.randomSplit(new double[]{0.9,0.1}, 123321);
        Dataset<Row> trainSet = datasets[0];
        Dataset<Row> testSet = datasets[1];


        GaussianMixture gmm = new GaussianMixture().setFeaturesCol("w2vRes").setK(2);
        GaussianMixtureModel model = gmm.fit(trainSet);

        try {
            model.save(outputDirectory);
        }
        catch (IOException e) {
            logger.log(Level.ERROR, e.getMessage());
        }

        // Shows the result.
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
