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
 * Created by steve on 09/02/2017.
 */
public class Word2Vector {
    public static void main(String[] args) {
        String inputDirectory = "src/main/resources/tfIdfData.json";
        String outputDirectory = "src/main/resources/w2vData";
        SparkSession spark = SparkSession.builder()
                .appName("IRProjectKMeans")
                .getOrCreate();
        Logger logger = LogManager.getRootLogger();
        logger.setLevel(Level.WARN);
        Dataset<Row> data = spark.read().parquet(inputDirectory);

        Word2Vec word2Vec = new Word2Vec()
                .setInputCol("woutSWords")
                .setOutputCol("w2vRes")
                .setVectorSize(50)
                .setMinCount(0);

        Word2VecModel model = word2Vec.fit(data);


        Dataset<Row> transformed =  model.transform(data);
        transformed.write().parquet(outputDirectory);

    }
}
