package gr.auth.csd.IRProject.supervised;

import org.apache.log4j.Level;
import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;
import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.ml.feature.LabeledPoint;
import org.apache.spark.mllib.classification.SVMWithSGD;
import org.apache.spark.ml.linalg.SparseVector;
import org.apache.spark.mllib.optimization.L1Updater;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import scala.Function1;
import scala.collection.immutable.List;

import java.util.ArrayList;
import java.util.Arrays;

/**
 * Created by steve on 03/01/2017.
 */
public class SVMAlgorithm {
    public static void main(String[] args) {
        String inputDirectory = "src/main/resources/tfIdfData.json";
        String ioSVMDirectory = "src/main/resources/tfIdfData.svm";
        String outputDirectory = "src/main/resources/svmModel";
        SparkSession spark = SparkSession.builder()
                .appName("IRProjectKMeans")
                .getOrCreate();
        Logger logger = LogManager.getRootLogger();
        logger.setLevel(Level.WARN);
        Dataset<Row> data = spark.read().parquet(inputDirectory);

        Dataset<String> ds = data.select("label", "features").map((MapFunction)(Object r) -> {
            Row r1 = (Row) r;
            String label = Long.toString(r1.getLong(0));
            SparseVector sparseVector = (SparseVector) r1.get(1);
            ArrayList<String> toReturn = new ArrayList<>();
            int[] indices = sparseVector.indices();
            double[] values = sparseVector.values();
            for(int i=0;i<indices.length;++i) {
                toReturn.add(String.format("%d:%f", indices[i], values[i]));
            }
            return label + " " + String.join(" ", toReturn);
        }, Encoders.STRING());
        ds.write().text(ioSVMDirectory);

//        Dataset<Row>[] datasets = data.randomSplit(new double[]{0.9,0.1}, 123321);
//        Dataset<Row> trainSet = datasets[0];
//        Dataset<Row> testSet = datasets[1];
//
//        SVMWithSGD svm = new SVMWithSGD();
//        svm.optimizer().setNumIterations(100).setRegParam(0.05).setUpdater(new L1Updater());
//        svm.run(trainSet.rdd());
    }
}
