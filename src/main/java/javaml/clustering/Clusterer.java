package javaml.clustering;

public interface Clusterer {

    void fit(double[][] X);

    int[] getLabels();
}