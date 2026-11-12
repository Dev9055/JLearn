package javaml.regression;

public class LassoRegression implements Regressor {

    private double[] weights;
    private double bias;

    private double learningRate;
    private int iterations;
    private double lambda;

    public LassoRegression() {
        learningRate = 0.01;
        iterations = 1000;
        lambda = 0.01;
    }

    public LassoRegression(
            double learningRate,
            int iterations,
            double lambda) {

        this.learningRate = learningRate;
        this.iterations = iterations;
        this.lambda = lambda;
    }

    @Override
    public void fit(double[][] X, double[] y) {

        int rows = X.length;
        int features = X[0].length;

        weights = new double[features];
        bias = 0;

        for (int iteration = 0; iteration < iterations; iteration++) {

            double[] predictions = predict(X);

            double[] weightGradient = new double[features];

            double biasGradient = 0;

            for (int i = 0; i < rows; i++) {

                double error = predictions[i] - y[i];

                biasGradient += error;

                for (int j = 0; j < features; j++) {

                    weightGradient[j] += error * X[i][j];
                }
            }

            for (int j = 0; j < features; j++) {

                weightGradient[j] = weightGradient[j] / rows;

                if (weights[j] > 0) {
                    weightGradient[j] += lambda;
                } else if (weights[j] < 0) {
                    weightGradient[j] -= lambda;
                }

                weights[j] = weights[j]
                        - learningRate
                                * weightGradient[j];
            }

            biasGradient = biasGradient / rows;

            bias = bias
                    - learningRate
                            * biasGradient;
        }
    }

    @Override
    public double[] predict(double[][] X) {

        double[] predictions = new double[X.length];

        for (int i = 0; i < X.length; i++) {

            double prediction = bias;

            for (int j = 0; j < X[i].length; j++) {

                prediction += X[i][j] * weights[j];
            }

            predictions[i] = prediction;
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