package javaml.data;

public class Dataset {

    private double[][] X;
    private double[] y;

    public Dataset(double[][] X, double[] y) {

        if (X == null || y == null) {
            throw new IllegalArgumentException(
                    "X and y cannot be null.");
        }

        if (X.length == 0) {
            throw new IllegalArgumentException(
                    "Dataset cannot be empty.");
        }

        if (X.length != y.length) {
            throw new IllegalArgumentException(
                    "X and y must have the same number of rows.");
        }

        if (X[0].length == 0) {
            throw new IllegalArgumentException(
                    "Dataset must contain at least one feature.");
        }

        this.X = X;
        this.y = y;
    }

    public double[][] getX() {
        return X;
    }

    public double[] getY() {
        return y;
    }

    public int getRows() {
        return X.length;
    }

    public int getFeatures() {
        return X[0].length;
    }
}