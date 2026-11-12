package javaml.regression;

public class LinearRegression implements Regressor {

    private double[] weights;
    private double bias;

    private double learningRate;
    private int iterations;

    public LinearRegression() {
        learningRate = 0.01;
        iterations = 1000;
    }

    public LinearRegression(double learningRate, int iterations) {
        this.learningRate = learningRate;
        this.iterations = iterations;
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