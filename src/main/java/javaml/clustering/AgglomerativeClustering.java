package javaml.clustering;

import java.util.ArrayList;

public class AgglomerativeClustering
        implements Clusterer {

    private int k;
    private int[] labels;

    public AgglomerativeClustering(int k) {
        this.k = k;
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

    private double clusterDistance(
            ArrayList<Integer> cluster1,
            ArrayList<Integer> cluster2,
            double[][] X) {

        double minimum = Double.MAX_VALUE;

        for (int i : cluster1) {

            for (int j : cluster2) {

                double currentDistance = distance(
                        X[i],
                        X[j]);

                if (currentDistance < minimum) {

                    minimum = currentDistance;
                }
            }
        }

        return minimum;
    }

    @Override
    public void fit(double[][] X) {

        ArrayList<ArrayList<Integer>> clusters = new ArrayList<>();

        // Every point starts as its own cluster

        for (int i = 0; i < X.length; i++) {

            ArrayList<Integer> cluster = new ArrayList<>();

            cluster.add(i);

            clusters.add(cluster);
        }

        while (clusters.size() > k) {

            double bestDistance = Double.MAX_VALUE;

            int first = -1;
            int second = -1;

            for (int i = 0; i < clusters.size(); i++) {

                for (int j = i + 1; j < clusters.size(); j++) {

                    double currentDistance = clusterDistance(
                            clusters.get(i),
                            clusters.get(j),
                            X);

                    if (currentDistance < bestDistance) {

                        bestDistance = currentDistance;

                        first = i;
                        second = j;
                    }
                }
            }

            clusters.get(first)
                    .addAll(
                            clusters.get(second));

            clusters.remove(second);
        }

        labels = new int[X.length];

        for (int c = 0; c < clusters.size(); c++) {

            for (int index : clusters.get(c)) {

                labels[index] = c;
            }
        }
    }

    @Override
    public int[] getLabels() {
        return labels;
    }
}