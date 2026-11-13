package javaml;

import javaml.data.CSVReader;

import javaml.preprocessing.StandardScaler;

import javaml.clustering.*;

import javaml.metrics.ClusteringMetrics;

public class ClusteringTest {

    public static void main(String[] args) {

        double[][] X = CSVReader.readFeatures(
                "datasets/customers.csv");

        StandardScaler scaler = new StandardScaler();

        X = scaler.fitTransform(X);

        Clusterer[] models = {

                new KMeans(2),

                new KMeansPlusPlus(2),

                new KMedoids(2),

                new AgglomerativeClustering(2),

                new MeanShift(1.0),

                new DBSCAN(0.5, 3)
        };

        String[] names = {

                "K-Means",
                "K-Means++",
                "K-Medoids",
                "Agglomerative",
                "Mean Shift",
                "DBSCAN"
        };

        for (int i = 0; i < models.length; i++) {

            models[i].fit(X);

            int[] labels = models[i].getLabels();

            double score = ClusteringMetrics
                    .silhouetteScore(
                            X,
                            labels);

            System.out.println(
                    names[i]
                            + " -> Silhouette: "
                            + score);
        }
    }
}