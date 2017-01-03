package gr.auth.csd.IRProject.unsupervised;

import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.linalg.Vector;
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
        for(int i=0;i<10;++i) {
            Dataset<Row>[] splitDataset = data.randomSplit(new double[]{0.9, 0.1});
            KMeans kmeans = new KMeans().setK(2).setSeed(2L);
            KMeansModel model = kmeans.fit(splitDataset[0]);
        }



        double WSSSE = model.computeCost(data);
        System.out.println("Within Set Sum of Squared Errors = " + WSSSE);

        // Shows the result.
        Vector[] centers = model.clusterCenters();
        System.out.println("Cluster Centers: ");
        for (Vector center: centers) {
            System.out.println(center);
        }

        spark.stop();
    }
}
