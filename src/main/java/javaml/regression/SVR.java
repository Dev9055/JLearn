package javaml.regression;

public class SVR implements Regressor {

    private double[] weights;
    private double bias;

    private double learningRate;
    private double lambda;
    private double epsilon;
    private int iterations;

    public SVR() {

        learningRate = 0.001;
        lambda = 0.01;
        epsilon = 0.1;
        iterations = 1000;
    }

    public SVR(
            double learningRate,
            double lambda,
            double epsilon,
            int iterations) {

        this.learningRate = learningRate;
        this.lambda = lambda;
        this.epsilon = epsilon;
        this.iterations = iterations;
    }

    @Override
    public void fit(double[][] X, double[] y) {

        int features = X[0].length;

        weights = new double[features];

        bias = 0;

        for (int iteration = 0; iteration < iterations; iteration++) {

            for (int i = 0; i < X.length; i++) {

                double prediction = bias;

                for (int j = 0; j < features; j++) {

                    prediction += weights[j]
                            * X[i][j];
                }

                double error = prediction - y[i];

                if (Math.abs(error) <= epsilon) {

                    for (int j = 0; j < features; j++) {

                        weights[j] -= learningRate
                                * 2
                                * lambda
                                * weights[j];
                    }
                }

                else {

                    double direction;

                    if (error > 0) {
                        direction = 1;
                    } else {
                        direction = -1;
                    }

                    for (int j = 0; j < features; j++) {

                        weights[j] -= learningRate
                                * (2
                                        * lambda
                                        * weights[j]
                                        +
                                        direction
                                                * X[i][j]);
                    }

                    bias -= learningRate
                            * direction;
                }
            }
        }
    }

    @Override
    public double[] predict(double[][] X) {

        double[] predictions = new double[X.length];

        for (int i = 0; i < X.length; i++) {

            double prediction = bias;

            for (int j = 0; j < X[i].length; j++) {

                prediction += weights[j]
                        * X[i][j];
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