import javaml.clustering.Clusterer;
import javaml.clustering.KMeans;

public class ClusteringExample {

    public static void main(String[] args) {

        double[][] X = {

                { 1, 1 },
                { 1.5, 2 },
                { 2, 1 },

                { 8, 8 },
                { 9, 8 },
                { 8, 9 }
        };

        Clusterer model = new KMeans(2);

        model.fit(X);

        int[] labels = model.getLabels();

        for (int i = 0; i < labels.length; i++) {

            System.out.println(
                    "Point "
                            + (i + 1)
                            + " -> Cluster "
                            + labels[i]);
        }
    }
}