package javaml.clustering;

import java.util.Random;

public class KMeansPlusPlus extends KMeans {

    public KMeansPlusPlus(int k) {
        super(k);
    }

    public KMeansPlusPlus(
            int k,
            int maxIterations) {

        super(k, maxIterations);
    }

    @Override
    protected void initializeCentroids(double[][] X) {

        Random random = new Random();

        centroids = new double[k][X[0].length];

        int first = random.nextInt(X.length);

        for (int j = 0; j < X[0].length; j++) {

            centroids[0][j] = X[first][j];
        }

        for (int c = 1; c < k; c++) {

            double[] distances = new double[X.length];

            double totalDistance = 0;

            for (int i = 0; i < X.length; i++) {

                double minDistance = Double.MAX_VALUE;

                for (int existing = 0; existing < c; existing++) {

                    double d = distance(
                            X[i],
                            centroids[existing]);

                    double squaredDistance = d * d;

                    if (squaredDistance < minDistance) {

                        minDistance = squaredDistance;
                    }
                }

                distances[i] = minDistance;

                totalDistance += minDistance;
            }

            double randomValue = random.nextDouble()
                    * totalDistance;

            double sum = 0;

            int selectedIndex = 0;

            for (int i = 0; i < X.length; i++) {

                sum += distances[i];

                if (sum >= randomValue) {

                    selectedIndex = i;
                    break;
                }
            }

            for (int j = 0; j < X[0].length; j++) {

                centroids[c][j] = X[selectedIndex][j];
            }
        }
    }
}