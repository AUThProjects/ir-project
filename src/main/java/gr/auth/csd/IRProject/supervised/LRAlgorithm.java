package gr.auth.csd.IRProject.supervised;

import org.apache.log4j.Level;
import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.util.Arrays;

/**
 * Logistic regression wrapper class
 */
public class LRAlgorithm {
    /**
     * The output directory of the trained model.
     */
    static String outputModelDirectory = "src/main/resources/lrModel";
    static String inputLabelledDirectory = "src/main/resources/tfIdfData.json";
    static String inputUnlabelledDirectory = "src/main/resources/tfIdfDataUnlabelled.json";
    static String outputDataDirectory = "src/main/resources/predictions.txt";
    static Logger logger;

    static {
        logger = LogManager.getRootLogger();
        logger.setLevel(Level.WARN);
    }

    /**
     * Execution entrypoint.
     * @param args CLI arguments
     */
    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder()
                .appName("IRProjectLogisticRegression")
                .getOrCreate();


        Dataset<Row> labelledData = spark.read().parquet(inputLabelledDirectory);
        train(labelledData, new double[]{0.9,0.1}, 123321);

        Dataset<Row> unlabelledData = spark.read().parquet(inputUnlabelledDirectory);
        test(unlabelledData, outputDataDirectory);
    }

    /**
     * Method training the model using Cross Validation.
     * @param data The labelled data to be trained on
     * @param split Percentages split on holdout
     * @param seed Seed for randomness
     */
    public static void train(Dataset<Row> data, double[] split, int seed) {
        Dataset<Row>[] datasets = data.randomSplit(split, seed);
        Dataset<Row> trainSet = datasets[0];
        Dataset<Row> testSet = datasets[1];

        LogisticRegression lr = new LogisticRegression().setPredictionCol("prediction").setFeaturesCol("features").setMaxIter(200);
        ParamMap[] paramGrid = new ParamGridBuilder().build();


        CrossValidator cv = new CrossValidator()
                .setEstimator(lr.setRegParam(0.05))
                .setEvaluator(new BinaryClassificationEvaluator().setRawPredictionCol("prediction"))
                .setEstimatorParamMaps(paramGrid).setNumFolds(3);

        CrossValidatorModel cvModel = cv.fit(trainSet);

        Dataset<Row> predictions = cvModel.transform(testSet);

        logger.log(Level.WARN, Arrays.toString(cvModel.avgMetrics()));


        // Now train on the whole dataset
        cvModel = cv.fit(data);

        try {
            cvModel.save(outputModelDirectory);
        }
        catch (java.io.IOException e) {
            logger.log(Level.ERROR, e.getMessage());
        }
    }

    /**
     * Method predicting labels on dataset.
     * @param data Tha unlabelled data to label
     * @param outputDataDirectory The output data directory
     */
    public static void test(Dataset<Row> data, String outputDataDirectory) {
        CrossValidatorModel cvModel = CrossValidatorModel.load(outputModelDirectory);
        Dataset<Row> predictions = cvModel.transform(data);

        predictions.select("id", "prediction").toJavaRDD().coalesce(1).saveAsTextFile(outputDataDirectory);
    }
}
