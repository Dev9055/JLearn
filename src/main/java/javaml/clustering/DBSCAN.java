package javaml.clustering;

import java.util.ArrayList;

public class DBSCAN implements Clusterer {

    private double eps;
    private int minPoints;

    private int[] labels;
    private boolean[] visited;

    public DBSCAN(
            double eps,
            int minPoints) {

        this.eps = eps;
        this.minPoints = minPoints;
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

    private ArrayList<Integer> findNeighbors(
            double[][] X,
            int point) {

        ArrayList<Integer> neighbors = new ArrayList<>();

        for (int i = 0; i < X.length; i++) {

            if (distance(
                    X[point],
                    X[i]) <= eps) {

                neighbors.add(i);
            }
        }

        return neighbors;
    }

    private void expandCluster(
            double[][] X,
            int point,
            ArrayList<Integer> neighbors,
            int cluster) {

        labels[point] = cluster;

        for (int i = 0; i < neighbors.size(); i++) {

            int current = neighbors.get(i);

            if (!visited[current]) {

                visited[current] = true;

                ArrayList<Integer> newNeighbors = findNeighbors(
                        X,
                        current);

                if (newNeighbors.size() >= minPoints) {

                    for (int neighbor : newNeighbors) {

                        if (!neighbors.contains(neighbor)) {
                            neighbors.add(neighbor);
                        }
                    }
                }
            }

            if (labels[current] == -1) {

                labels[current] = cluster;
            }
        }
    }

    @Override
    public void fit(double[][] X) {

        labels = new int[X.length];
        visited = new boolean[X.length];

        for (int i = 0; i < labels.length; i++) {
            labels[i] = -1;
        }

        int cluster = 0;

        for (int i = 0; i < X.length; i++) {

            if (visited[i]) {
                continue;
            }

            visited[i] = true;

            ArrayList<Integer> neighbors = findNeighbors(
                    X,
                    i);

            if (neighbors.size() < minPoints) {

                labels[i] = -1;
            }

            else {

                expandCluster(
                        X,
                        i,
                        neighbors,
                        cluster);

                cluster++;
            }
        }
    }

    @Override
    public int[] getLabels() {
        return labels;
    }
}