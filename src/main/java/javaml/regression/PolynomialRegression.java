package javaml.regression;

public class PolynomialRegression implements Regressor {

    private int degree;
    private LinearRegression model;

    public PolynomialRegression(int degree) {
        this.degree = degree;
        model = new LinearRegression();
    }

    public PolynomialRegression(
            int degree,
            double learningRate,
            int iterations) {

        this.degree = degree;

        model = new LinearRegression(
                learningRate,
                iterations);
    }

    private double[][] transform(double[][] X) {

        int rows = X.length;
        int features = X[0].length;

        double[][] result = new double[rows][features * degree];

        for (int i = 0; i < rows; i++) {

            int column = 0;

            for (int j = 0; j < features; j++) {

                for (int d = 1; d <= degree; d++) {

                    result[i][column] = Math.pow(X[i][j], d);

                    column++;
                }
            }
        }

        return result;
    }

    @Override
    public void fit(double[][] X, double[] y) {

        double[][] polynomialX = transform(X);

        model.fit(polynomialX, y);
    }

    @Override
    public double[] predict(double[][] X) {

        double[][] polynomialX = transform(X);

        return model.predict(polynomialX);
    }
}