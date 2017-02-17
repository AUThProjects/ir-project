package gr.auth.csd.IRProject.supervised;

import org.apache.log4j.Level;
import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.classification.SVMModel;
import org.apache.spark.mllib.classification.SVMWithSGD;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.optimization.L1Updater;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;
import org.apache.spark.sql.SparkSession;
import scala.Tuple2;

/**
 * Trains and uses a Support Vector Machine in order to predict movie reviews' classes (positive, negative).
 * Requires its training set to be loaded in libsvm format.
 *
 * For more information, please refer to <a href="https://www.csie.ntu.edu.tw/~cjlin/libsvm/">libsvm website</a>.
 */
public class SVMAlgorithm {
    static String ioSVMDirectory = "src/main/resources/tfIdfData.svm";
    static String outputDirectory = "src/main/resources/svmModel";
    static Logger logger;
    static {
        logger = LogManager.getRootLogger();
        logger.setLevel(Level.WARN);
    }
    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder()
                                         .appName("IRProjectSVM")
                                         .getOrCreate();

        JavaRDD<LabeledPoint> svmData = MLUtils.loadLibSVMFile(spark.sparkContext(), ioSVMDirectory).toJavaRDD();

        // 90-10% hold-out validation
        JavaRDD<LabeledPoint>[] datasets = svmData.randomSplit(new double[]{0.9,0.1}, 123321);
        JavaRDD<LabeledPoint> trainSet = datasets[0];
        JavaRDD<LabeledPoint> testSet = datasets[1];

        // Train SVM.
        SVMWithSGD svm = new SVMWithSGD();
        svm.optimizer().setNumIterations(100).setRegParam(0.05).setUpdater(new L1Updater());
        SVMModel model = svm.run(trainSet.rdd());
        model.clearThreshold();

        // Save SVM model.
        model.save(spark.sparkContext(), outputDirectory);

        // Compute raw scores on the test set.
        JavaRDD<Tuple2<Object, Object>> scoreAndLabels = testSet.map(
                new Function<LabeledPoint, Tuple2<Object, Object>>() {
                    public Tuple2<Object, Object> call(LabeledPoint p) {
                        Double score = model.predict(p.features());
                        return new Tuple2<Object, Object>(score, p.label());
                    }
                }
        );

        // Get evaluation metrics.
        BinaryClassificationMetrics metrics = new BinaryClassificationMetrics(JavaRDD.toRDD(scoreAndLabels));
        double auROC = metrics.areaUnderROC();

        logger.log(Level.WARN, "Area under ROC = " + auROC);
    }
}
