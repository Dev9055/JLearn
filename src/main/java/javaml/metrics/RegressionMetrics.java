package javaml.metrics;

public class RegressionMetrics {

    public static double mae(
            double[] actual,
            double[] predicted) {

        double sum = 0;

        for (int i = 0; i < actual.length; i++) {

            sum += Math.abs(
                    actual[i] - predicted[i]);
        }

        return sum / actual.length;
    }

    public static double mse(
            double[] actual,
            double[] predicted) {

        double sum = 0;

        for (int i = 0; i < actual.length; i++) {

            double error = actual[i] - predicted[i];

            sum += error * error;
        }

        return sum / actual.length;
    }

    public static double rmse(
            double[] actual,
            double[] predicted) {

        return Math.sqrt(
                mse(actual, predicted));
    }

    public static double r2Score(
            double[] actual,
            double[] predicted) {

        double mean = 0;

        for (double value : actual) {
            mean += value;
        }

        mean = mean / actual.length;

        double totalSum = 0;
        double residualSum = 0;

        for (int i = 0; i < actual.length; i++) {

            double totalDifference = actual[i] - mean;

            totalSum += totalDifference
                    * totalDifference;

            double predictionDifference = actual[i] - predicted[i];

            residualSum += predictionDifference
                    * predictionDifference;
        }

        if (totalSum == 0) {
            return 0;
        }

        return 1 -
                (residualSum / totalSum);
    }
}