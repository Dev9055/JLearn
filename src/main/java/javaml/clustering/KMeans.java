package javaml.clustering;

import java.util.Random;

public class KMeans implements Clusterer {

    protected int k;
    protected int maxIterations;

    protected double[][] centroids;
    protected int[] labels;

    public KMeans(int k) {
        this.k = k;
        this.maxIterations = 100;
    }

    public KMeans(int k, int maxIterations) {
        this.k = k;
        this.maxIterations = maxIterations;
    }

    protected double distance(
            double[] a,
            double[] b) {

        double sum = 0;

        for (int i = 0; i < a.length; i++) {

            double difference = a[i] - b[i];

            sum += difference * difference;
        }

        return Math.sqrt(sum);
    }

    protected void initializeCentroids(double[][] X) {

        Random random = new Random();

        centroids = new double[k][X[0].length];

        for (int i = 0; i < k; i++) {

            int index = random.nextInt(X.length);

            for (int j = 0; j < X[0].length; j++) {

                centroids[i][j] = X[index][j];
            }
        }
    }

    @Override
    public void fit(double[][] X) {

        initializeCentroids(X);

        labels = new int[X.length];

        for (int iteration = 0; iteration < maxIterations; iteration++) {

            boolean changed = false;

            // Assign points to nearest centroid

            for (int i = 0; i < X.length; i++) {

                int bestCluster = 0;

                double bestDistance = distance(
                        X[i],
                        centroids[0]);

                for (int c = 1; c < k; c++) {

                    double currentDistance = distance(
                            X[i],
                            centroids[c]);

                    if (currentDistance < bestDistance) {

                        bestDistance = currentDistance;

                        bestCluster = c;
                    }
                }

                if (labels[i] != bestCluster) {

                    labels[i] = bestCluster;
                    changed = true;
                }
            }

            // Calculate new centroids

            double[][] newCentroids = new double[k][X[0].length];

            int[] counts = new int[k];

            for (int i = 0; i < X.length; i++) {

                int cluster = labels[i];

                counts[cluster]++;

                for (int j = 0; j < X[i].length; j++) {

                    newCentroids[cluster][j] += X[i][j];
                }
            }

            for (int c = 0; c < k; c++) {

                if (counts[c] == 0) {
                    continue;
                }

                for (int j = 0; j < X[0].length; j++) {

                    newCentroids[c][j] /= counts[c];
                }
            }

            centroids = newCentroids;

            if (!changed) {
                break;
            }
        }
    }

    @Override
    public int[] getLabels() {
        return labels;
    }

    public double[][] getCentroids() {
        return centroids;
    }
}