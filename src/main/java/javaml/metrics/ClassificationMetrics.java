package javaml.metrics;

public class ClassificationMetrics {

    public static double accuracy(
            double[] actual,
            int[] predicted) {

        int correct = 0;

        for (int i = 0; i < actual.length; i++) {

            if ((int) actual[i] == predicted[i]) {
                correct++;
            }
        }

        return (double) correct / actual.length;
    }

    public static double precision(
            double[] actual,
            int[] predicted) {

        int truePositive = 0;
        int falsePositive = 0;

        for (int i = 0; i < actual.length; i++) {

            if (predicted[i] == 1) {

                if ((int) actual[i] == 1) {
                    truePositive++;
                } else {
                    falsePositive++;
                }
            }
        }

        if (truePositive + falsePositive == 0) {
            return 0;
        }

        return (double) truePositive
                / (truePositive + falsePositive);
    }

    public static double recall(
            double[] actual,
            int[] predicted) {

        int truePositive = 0;
        int falseNegative = 0;

        for (int i = 0; i < actual.length; i++) {

            if ((int) actual[i] == 1) {

                if (predicted[i] == 1) {
                    truePositive++;
                } else {
                    falseNegative++;
                }
            }
        }

        if (truePositive + falseNegative == 0) {
            return 0;
        }

        return (double) truePositive
                / (truePositive + falseNegative);
    }

    public static double f1Score(
            double[] actual,
            int[] predicted) {

        double precision = precision(actual, predicted);

        double recall = recall(actual, predicted);

        if (precision + recall == 0) {
            return 0;
        }

        return 2
                * precision
                * recall
                / (precision + recall);
    }
}