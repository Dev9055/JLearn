package javaml.classification;

public class Perceptron implements Classifier {

    private double[] weights;
    private double bias;

    private double learningRate;
    private int iterations;

    public Perceptron() {
        learningRate = 0.01;
        iterations = 1000;
    }

    public Perceptron(
            double learningRate,
            int iterations) {

        this.learningRate = learningRate;
        this.iterations = iterations;
    }

    @Override
    public void fit(double[][] X, double[] y) {

        int features = X[0].length;

        weights = new double[features];
        bias = 0;

        for (int iteration = 0; iteration < iterations; iteration++) {

            for (int i = 0; i < X.length; i++) {

                double value = bias;

                for (int j = 0; j < features; j++) {

                    value += X[i][j] * weights[j];
                }

                int prediction;

                if (value >= 0) {
                    prediction = 1;
                } else {
                    prediction = 0;
                }

                double error = y[i] - prediction;

                for (int j = 0; j < features; j++) {

                    weights[j] += learningRate
                            * error
                            * X[i][j];
                }

                bias += learningRate * error;
            }
        }
    }

    @Override
    public int[] predict(double[][] X) {

        int[] predictions = new int[X.length];

        for (int i = 0; i < X.length; i++) {

            double value = bias;

            for (int j = 0; j < X[i].length; j++) {

                value += X[i][j] * weights[j];
            }

            if (value >= 0) {
                predictions[i] = 1;
            } else {
                predictions[i] = 0;
            }
        }

        return predictions;
    }
}