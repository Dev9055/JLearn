package javaml.preprocessing;

public class StandardScaler {

    private double[] mean;
    private double[] std;

    public void fit(double[][] X) {

        int rows = X.length;
        int columns = X[0].length;

        mean = new double[columns];
        std = new double[columns];

        for (int j = 0; j < columns; j++) {

            double sum = 0;

            for (int i = 0; i < rows; i++) {
                sum += X[i][j];
            }

            mean[j] = sum / rows;
        }

        for (int j = 0; j < columns; j++) {

            double sum = 0;

            for (int i = 0; i < rows; i++) {
                double difference = X[i][j] - mean[j];
                sum += difference * difference;
            }

            std[j] = Math.sqrt(sum / rows);
        }
    }

    public double[][] transform(double[][] X) {

        int rows = X.length;
        int columns = X[0].length;

        double[][] result = new double[rows][columns];

        for (int i = 0; i < rows; i++) {

            for (int j = 0; j < columns; j++) {

                if (std[j] == 0) {
                    result[i][j] = 0;
                } else {
                    result[i][j] = (X[i][j] - mean[j]) / std[j];
                }
            }
        }

        return result;
    }

    public double[][] fitTransform(double[][] X) {

        fit(X);

        return transform(X);
    }
}