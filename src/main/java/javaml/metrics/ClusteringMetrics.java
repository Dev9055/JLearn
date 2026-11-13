package javaml.metrics;

public class ClusteringMetrics {

    private static double distance(
            double[] a,
            double[] b) {

        double sum = 0;

        for (int i = 0; i < a.length; i++) {

            double difference = a[i] - b[i];

            sum += difference * difference;
        }

        return Math.sqrt(sum);
    }

    public static double silhouetteScore(
            double[][] X,
            int[] labels) {

        if (X.length <= 1) {
            return 0;
        }

        double totalScore = 0;

        for (int i = 0; i < X.length; i++) {

            int currentCluster = labels[i];

            // DBSCAN noise point
            if (currentCluster == -1) {
                continue;
            }

            double sameClusterDistance = 0;
            int sameClusterCount = 0;

            for (int j = 0; j < X.length; j++) {

                if (i == j) {
                    continue;
                }

                if (labels[j] == currentCluster) {

                    sameClusterDistance += distance(
                            X[i],
                            X[j]);

                    sameClusterCount++;
                }
            }

            double a = 0;

            if (sameClusterCount > 0) {

                a = sameClusterDistance
                        / sameClusterCount;
            }

            double b = Double.MAX_VALUE;

            for (int cluster : labels) {

                if (cluster == currentCluster
                        || cluster == -1) {

                    continue;
                }

                double clusterDistance = 0;
                int clusterCount = 0;

                for (int j = 0; j < X.length; j++) {

                    if (labels[j] == cluster) {

                        clusterDistance += distance(
                                X[i],
                                X[j]);

                        clusterCount++;
                    }
                }

                if (clusterCount > 0) {

                    double averageDistance = clusterDistance
                            / clusterCount;

                    if (averageDistance < b) {

                        b = averageDistance;
                    }
                }
            }

            if (b == Double.MAX_VALUE) {
                continue;
            }

            double denominator = Math.max(a, b);

            double score = 0;

            if (denominator != 0) {

                score = (b - a)
                        / denominator;
            }

            totalScore += score;
        }

        return totalScore / X.length;
    }
}