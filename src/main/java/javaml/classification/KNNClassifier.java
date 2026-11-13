package javaml.classification;

public class KNNClassifier implements Classifier {

    private int k;

    private double[][] XTrain;
    private double[] yTrain;

    public KNNClassifier(int k) {
        this.k = k;
    }

    @Override
    public void fit(double[][] X, double[] y) {

        XTrain = X;
        yTrain = y;
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

    @Override
    public int[] predict(double[][] X) {

        int[] predictions = new int[X.length];

        for (int i = 0; i < X.length; i++) {

            double[] distances = new double[XTrain.length];

            int[] indexes = new int[XTrain.length];

            for (int j = 0; j < XTrain.length; j++) {

                distances[j] = distance(
                        X[i],
                        XTrain[j]);

                indexes[j] = j;
            }

            for (int a = 0; a < distances.length - 1; a++) {

                for (int b = a + 1; b < distances.length; b++) {

                    if (distances[a] > distances[b]) {

                        double tempDistance = distances[a];

                        distances[a] = distances[b];

                        distances[b] = tempDistance;

                        int tempIndex = indexes[a];

                        indexes[a] = indexes[b];

                        indexes[b] = tempIndex;
                    }
                }
            }

            int zeroCount = 0;
            int oneCount = 0;

            for (int j = 0; j < k; j++) {

                int label = (int) yTrain[indexes[j]];

                if (label == 0) {
                    zeroCount++;
                } else {
                    oneCount++;
                }
            }

            if (oneCount > zeroCount) {
                predictions[i] = 1;
            } else {
                predictions[i] = 0;
            }
        }

        return predictions;
    }
}