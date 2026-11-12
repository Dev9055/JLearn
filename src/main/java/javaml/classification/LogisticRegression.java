package javaml.classification;

public class LogisticRegression implements Classifier {

    private double[] weights;
    private double bias;

    private double learningRate;
    private int iterations;

    public LogisticRegression() {
        learningRate = 0.01;
        iterations = 1000;
    }

    public LogisticRegression(
            double learningRate,
            int iterations) {

        this.learningRate = learningRate;
        this.iterations = iterations;
    }

    private double sigmoid(double value) {

        return 1.0 / (1.0 + Math.exp(-value));
    }

    @Override
    public void fit(double[][] X, double[] y) {

        int rows = X.length;
        int features = X[0].length;

        weights = new double[features];
        bias = 0;

        for (int iteration = 0; iteration < iterations; iteration++) {

            double[] weightGradient = new double[features];

            double biasGradient = 0;

            for (int i = 0; i < rows; i++) {

                double linear = bias;

                for (int j = 0; j < features; j++) {

                    linear += X[i][j] * weights[j];
                }

                double prediction = sigmoid(linear);

                double error = prediction - y[i];

                for (int j = 0; j < features; j++) {

                    weightGradient[j] += error * X[i][j];
                }

                biasGradient += error;
            }

            for (int j = 0; j < features; j++) {

                weights[j] -= learningRate
                        * weightGradient[j]
                        / rows;
            }

            bias -= learningRate
                    * biasGradient
                    / rows;
        }
    }

    public double[] predictProbability(double[][] X) {

        double[] probabilities = new double[X.length];

        for (int i = 0; i < X.length; i++) {

            double linear = bias;

            for (int j = 0; j < X[i].length; j++) {

                linear += X[i][j] * weights[j];
            }

            probabilities[i] = sigmoid(linear);
        }

        return probabilities;
    }

    @Override
    public int[] predict(double[][] X) {

        double[] probabilities = predictProbability(X);

        int[] predictions = new int[X.length];

        for (int i = 0; i < probabilities.length; i++) {

            if (probabilities[i] >= 0.5) {
                predictions[i] = 1;
            } else {
                predictions[i] = 0;
            }
        }

        return predictions;
    }
}