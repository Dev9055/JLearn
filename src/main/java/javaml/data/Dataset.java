package javaml.data;

public class Dataset {

    private double[][] X;
    private double[] y;

    public Dataset(double[][] X, double[] y) {
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