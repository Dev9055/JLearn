package javaml.classification;

public class GaussianNaiveBayes implements Classifier {

    private double[][] mean;
    private double[][] variance;
    private double[] prior;

    @Override
    public void fit(double[][] X, double[] y) {

        int rows = X.length;
        int features = X[0].length;

        int classes = 2;

        mean = new double[classes][features];
        variance = new double[classes][features];
        prior = new double[classes];

        int[] count = new int[classes];

        for (int i = 0; i < rows; i++) {

            int label = (int) y[i];

            count[label]++;

            for (int j = 0; j < features; j++) {

                mean[label][j] += X[i][j];
            }
        }

        for (int c = 0; c < classes; c++) {

            for (int j = 0; j < features; j++) {

                mean[c][j] = mean[c][j] / count[c];
            }

            prior[c] = (double) count[c] / rows;
        }

        for (int i = 0; i < rows; i++) {

            int label = (int) y[i];

            for (int j = 0; j < features; j++) {

                double difference = X[i][j]
                        - mean[label][j];

                variance[label][j] += difference * difference;
            }
        }

        for (int c = 0; c < classes; c++) {

            for (int j = 0; j < features; j++) {

                variance[c][j] = variance[c][j]
                        / count[c];
            }
        }
    }

    private double gaussian(
            double value,
            double mean,
            double variance) {

        if (variance == 0) {
            variance = 0.000001;
        }

        double exponent = Math.exp(
                -Math.pow(value - mean, 2)
                        / (2 * variance));

        return (1.0
                / Math.sqrt(
                        2 * Math.PI * variance))
                * exponent;
    }

    @Override
    public int[] predict(double[][] X) {

        int[] predictions = new int[X.length];

        for (int i = 0; i < X.length; i++) {

            double bestProbability = -1;

            int bestClass = 0;

            for (int c = 0; c < 2; c++) {

                double probability = prior[c];

                for (int j = 0; j < X[i].length; j++) {

                    probability *= gaussian(
                            X[i][j],
                            mean[c][j],
                            variance[c][j]);
                }

                if (probability > bestProbability) {

                    bestProbability = probability;

                    bestClass = c;
                }
            }

            predictions[i] = bestClass;
        }

        return predictions;
    }
}