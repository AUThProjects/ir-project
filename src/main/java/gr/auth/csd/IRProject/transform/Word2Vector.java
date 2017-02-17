package gr.auth.csd.IRProject.transform;

import org.apache.log4j.Level;
import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;
import org.apache.spark.ml.feature.Word2Vec;
import org.apache.spark.ml.feature.Word2VecModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

/**
 * Word2Vec class for transforming features to concepts and reducing dimensions
 */
public class Word2Vector {

    static Logger logger;

    static {
        logger = LogManager.getRootLogger();
        logger.setLevel(Level.WARN);
    }

    /**
     * Execution entrypoint.
     * @param args CLI args
     */
    public static void main(String[] args) {
        String inputDirectory = "src/main/resources/tfIdfData.json";
        String outputDirectory = "src/main/resources/w2vData";
        final int vectorSize = 50;

        SparkSession spark = SparkSession.builder()
                .appName("IRProjectKMeans")
                .getOrCreate();
        Dataset<Row> data = spark.read().parquet(inputDirectory);

        Dataset<Row> transformed = word2vecTransform(spark, data, vectorSize);
        transformed.write().parquet(outputDirectory);
    }

    /**
     * Method responsible for doing the w2v transformation.
     * @param spark The spark session object
     * @param data The dataset to be transformed
     * @return The transformed dataset
     */
    public static Dataset<Row> word2vecTransform(SparkSession spark, Dataset<Row> data, int vectorSize) {
        Word2Vec word2Vec = new Word2Vec()
                .setInputCol("woutSWords")
                .setOutputCol("w2vRes")
                .setVectorSize(vectorSize)
                .setMinCount(0);
        Word2VecModel model = word2Vec.fit(data);
        return model.transform(data);
    }
}
