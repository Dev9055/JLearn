package javaml.clustering;

import java.util.ArrayList;

public class MeanShift implements Clusterer {

    private double bandwidth;
    private int maxIterations;

    private int[] labels;
    private double[][] centers;

    public MeanShift(double bandwidth) {

        this.bandwidth = bandwidth;
        this.maxIterations = 100;
    }

    public MeanShift(
            double bandwidth,
            int maxIterations) {

        this.bandwidth = bandwidth;
        this.maxIterations = maxIterations;
    }

    private double distance(
            double[] a,
            double[] b) {

        double sum = 0;

        for (int i = 0; i < a.length; i++) {

            double difference = a[i] - b[i];

            sum += difference * difference;
        }

        return Math.sqrt(sum);
    }

    private double[] shiftPoint(
            double[] point,
            double[][] X) {

        double[] current = point.clone();

        for (int iteration = 0; iteration < maxIterations; iteration++) {

            double[] mean = new double[current.length];

            int count = 0;

            for (int i = 0; i < X.length; i++) {

                if (distance(current, X[i]) <= bandwidth) {

                    for (int j = 0; j < current.length; j++) {

                        mean[j] += X[i][j];
                    }

                    count++;
                }
            }

            if (count == 0) {
                break;
            }

            for (int j = 0; j < mean.length; j++) {

                mean[j] /= count;
            }

            double movement = distance(current, mean);

            current = mean;

            if (movement < 0.001) {
                break;
            }
        }

        return current;
    }

    @Override
    public void fit(double[][] X) {

        double[][] shifted = new double[X.length][X[0].length];

        for (int i = 0; i < X.length; i++) {

            shifted[i] = shiftPoint(
                    X[i],
                    X);
        }

        ArrayList<double[]> centerList = new ArrayList<>();

        labels = new int[X.length];

        for (int i = 0; i < shifted.length; i++) {

            int cluster = -1;

            for (int c = 0; c < centerList.size(); c++) {

                if (distance(
                        shifted[i],
                        centerList.get(c)) < bandwidth / 2) {

                    cluster = c;
                    break;
                }
            }

            if (cluster == -1) {

                centerList.add(
                        shifted[i].clone());

                cluster = centerList.size() - 1;
            }

            labels[i] = cluster;
        }

        centers = new double[centerList.size()][];

        for (int i = 0; i < centerList.size(); i++) {

            centers[i] = centerList.get(i);
        }
    }

    @Override
    public int[] getLabels() {
        return labels;
    }

    public double[][] getCenters() {
        return centers;
    }

    public int getNumberOfClusters() {
        return centers.length;
    }
}