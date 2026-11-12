package javaml.clustering;

import java.util.Random;

public class KMedoids implements Clusterer {

    private int k;
    private int maxIterations;

    private double[][] medoids;
    private int[] labels;

    public KMedoids(int k) {
        this.k = k;
        this.maxIterations = 100;
    }

    public KMedoids(int k, int maxIterations) {
        this.k = k;
        this.maxIterations = maxIterations;
    }

    private double distance(double[] a, double[] b) {

        double sum = 0;

        for (int i = 0; i < a.length; i++) {

            double difference = a[i] - b[i];

            sum += difference * difference;
        }

        return Math.sqrt(sum);
    }

    private void initializeMedoids(double[][] X) {

        Random random = new Random();

        medoids = new double[k][X[0].length];

        boolean[] selected = new boolean[X.length];

        for (int i = 0; i < k; i++) {

            int index;

            do {

                index = random.nextInt(X.length);

            } while (selected[index]);

            selected[index] = true;

            for (int j = 0; j < X[0].length; j++) {

                medoids[i][j] = X[index][j];
            }
        }
    }

    @Override
    public void fit(double[][] X) {

        initializeMedoids(X);

        labels = new int[X.length];

        for (int iteration = 0; iteration < maxIterations; iteration++) {

            // Assign points to nearest medoid

            for (int i = 0; i < X.length; i++) {

                int bestCluster = 0;

                double bestDistance = distance(
                        X[i],
                        medoids[0]);

                for (int c = 1; c < k; c++) {

                    double currentDistance = distance(
                            X[i],
                            medoids[c]);

                    if (currentDistance < bestDistance) {

                        bestDistance = currentDistance;

                        bestCluster = c;
                    }
                }

                labels[i] = bestCluster;
            }

            boolean changed = false;

            // Find best medoid for each cluster

            for (int c = 0; c < k; c++) {

                double bestCost = Double.MAX_VALUE;

                int bestIndex = -1;

                for (int i = 0; i < X.length; i++) {

                    if (labels[i] != c) {
                        continue;
                    }

                    double cost = 0;

                    for (int j = 0; j < X.length; j++) {

                        if (labels[j] == c) {

                            cost += distance(
                                    X[i],
                                    X[j]);
                        }
                    }

                    if (cost < bestCost) {

                        bestCost = cost;
                        bestIndex = i;
                    }
                }

                if (bestIndex != -1) {

                    for (int j = 0; j < X[0].length; j++) {

                        if (medoids[c][j] != X[bestIndex][j]) {

                            changed = true;
                        }

                        medoids[c][j] = X[bestIndex][j];
                    }
                }
            }

            if (!changed) {
                break;
            }
        }

        // Final assignment

        for (int i = 0; i < X.length; i++) {

            int bestCluster = 0;

            double bestDistance = distance(
                    X[i],
                    medoids[0]);

            for (int c = 1; c < k; c++) {

                double currentDistance = distance(
                        X[i],
                        medoids[c]);

                if (currentDistance < bestDistance) {

                    bestDistance = currentDistance;

                    bestCluster = c;
                }
            }

            labels[i] = bestCluster;
        }
    }

    @Override
    public int[] getLabels() {
        return labels;
    }

    public double[][] getMedoids() {
        return medoids;
    }
}