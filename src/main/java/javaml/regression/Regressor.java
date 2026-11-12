package javaml.regression;

public interface Regressor {

    void fit(double[][] X, double[] y);

    double[] predict(double[][] X);
}