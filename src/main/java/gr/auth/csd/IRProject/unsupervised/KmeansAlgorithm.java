package gr.auth.csd.IRProject.unsupervised;

import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class KmeansAlgorithm {
    public static void main(String[] args) {
        String inputDirectory = "src/main/resources/tfIdfData.json";
        SparkSession spark = SparkSession.builder()
                                         .appName("IRProjectKMeans")
                                         .getOrCreate();
        Dataset<Row> data = spark.read().parquet(inputDirectory);
        KMeans kmeans = new KMeans().setK(2).setSeed(2L);
        KMeansModel model = kmeans.fit(data);
        ParamMap[] paramGrid = new ParamGridBuilder().build();

        CrossValidator cv = new CrossValidator()
                .setEstimator(kmeans.setFeaturesCol("features").setPredictionCol("prediction"))
                .setEvaluator(new BinaryClassificationEvaluator().setRawPredictionCol("prediction"))
                .setEstimatorParamMaps(paramGrid).setNumFolds(10);  // Use 3+ in practice

        CrossValidatorModel cvModel = cv.fit(data);

        System.out.println(cvModel.avgMetrics());


//        double WSSSE = model.computeCost(data);
//        System.out.println("Within Set Sum of Squared Errors = " + WSSSE);
//
//        // Shows the result.
//        Vector[] centers = model.clusterCenters();
//        System.out.println("Cluster Centers: ");
//        for (Vector center: centers) {
//            System.out.println(center);
//        }

        spark.stop();
    }
}
