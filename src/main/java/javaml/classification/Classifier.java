package javaml.classification;

public interface Classifier {

    void fit(double[][] X, double[] y);

    int[] predict(double[][] X);
}