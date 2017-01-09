package gr.auth.csd.IRProject.supervised;

import org.apache.log4j.Level;
import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;
import org.apache.spark.ml.Model;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.mllib.classification.SVMWithSGD;
import org.apache.spark.mllib.optimization.L1Updater;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;

import java.util.Arrays;

/**
 * Created by steve on 03/01/2017.
 */
public class LRAlgorithm {
    public static void main(String[] args) {
        String inputDirectory = "src/main/resources/tfIdfData.json";
        String outputDirectory = "src/main/resources/lrModel";
        SparkSession spark = SparkSession.builder()
                .appName("IRProjectKMeans")
                .getOrCreate();
        Logger logger = LogManager.getRootLogger();
        logger.setLevel(Level.WARN);
        Dataset<Row> data = spark.read().parquet(inputDirectory);
        
        Dataset<Row>[] datasets = data.randomSplit(new double[]{0.9,0.1}, 123321);
        Dataset<Row> trainSet = datasets[0];
        Dataset<Row> testSet = datasets[1];

        LogisticRegression lr = new LogisticRegression().setPredictionCol("prediction").setFeaturesCol("features").setMaxIter(200);
        ParamMap[] paramGrid = new ParamGridBuilder().build();
//        ParamMap[] paramGrid = new ParamGridBuilder()
//                .addGrid(lr.regParam(), new double[] {0.1, 0.01})
//                .addGrid(lr.elasticNetParam(), new double[] {0.1, 0.01})
//                .build();


        CrossValidator cv = new CrossValidator()
                .setEstimator(lr.setRegParam(0.05))
                .setEvaluator(new BinaryClassificationEvaluator().setRawPredictionCol("prediction"))
                .setEstimatorParamMaps(paramGrid).setNumFolds(3);  // Use 3+ in practice

        CrossValidatorModel cvModel = cv.fit(trainSet);

        try {
            cvModel.save(outputDirectory);
        }
        catch (java.io.IOException e) {
            logger.log(Level.ERROR, e.getMessage());
        }

        Dataset<Row> predictions = cvModel.transform(testSet);
        predictions.show(1000);


        logger.log(Level.WARN, Arrays.toString(cvModel.avgMetrics()));
    }
}
