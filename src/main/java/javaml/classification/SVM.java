package javaml.classification;

public class SVM implements Classifier {

    private double[] weights;
    private double bias;

    private double learningRate;
    private double lambda;
    private int iterations;

    public SVM() {
        learningRate = 0.001;
        lambda = 0.01;
        iterations = 1000;
    }

    public SVM(
            double learningRate,
            double lambda,
            int iterations) {

        this.learningRate = learningRate;
        this.lambda = lambda;
        this.iterations = iterations;
    }

    @Override
    public void fit(double[][] X, double[] y) {

        int rows = X.length;
        int features = X[0].length;

        weights = new double[features];
        bias = 0;

        for (int iteration = 0; iteration < iterations; iteration++) {

            for (int i = 0; i < rows; i++) {

                double label;

                if (y[i] == 0) {
                    label = -1;
                } else {
                    label = 1;
                }

                double value = bias;

                for (int j = 0; j < features; j++) {

                    value += weights[j] * X[i][j];
                }

                if (label * value >= 1) {

                    for (int j = 0; j < features; j++) {

                        weights[j] -= learningRate
                                * (2 * lambda
                                        * weights[j]);
                    }
                }

                else {

                    for (int j = 0; j < features; j++) {

                        weights[j] -= learningRate
                                * (2 * lambda
                                        * weights[j]
                                        -
                                        label * X[i][j]);
                    }

                    bias += learningRate * label;
                }
            }
        }
    }

    @Override
    public int[] predict(double[][] X) {

        int[] predictions = new int[X.length];

        for (int i = 0; i < X.length; i++) {

            double value = bias;

            for (int j = 0; j < X[i].length; j++) {

                value += weights[j] * X[i][j];
            }

            if (value >= 0) {
                predictions[i] = 1;
            } else {
                predictions[i] = 0;
            }
        }

        return predictions;
    }

    public double[] getWeights() {
        return weights;
    }

    public double getBias() {
        return bias;
    }
}