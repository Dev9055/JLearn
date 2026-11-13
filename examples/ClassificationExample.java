import javaml.classification.Classifier;
import javaml.classification.LogisticRegression;

import javaml.metrics.ClassificationMetrics;

public class ClassificationExample {

    public static void main(String[] args) {

        double[][] X = {
                { 1, 20 },
                { 2, 30 },
                { 3, 40 },
                { 6, 70 },
                { 7, 80 },
                { 8, 90 }
        };

        double[] y = {
                0,
                0,
                0,
                1,
                1,
                1
        };

        Classifier model = new LogisticRegression(
                0.01,
                3000);

        model.fit(X, y);

        int[] predictions = model.predict(X);

        System.out.println(
                "Accuracy: "
                        + ClassificationMetrics.accuracy(
                                y,
                                predictions));
    }
}