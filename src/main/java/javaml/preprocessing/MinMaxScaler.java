package javaml.preprocessing;

public class MinMaxScaler {

    private double[] min;
    private double[] max;

    public void fit(double[][] X) {

        int rows = X.length;
        int columns = X[0].length;

        min = new double[columns];
        max = new double[columns];

        for (int j = 0; j < columns; j++) {

            min[j] = X[0][j];
            max[j] = X[0][j];

            for (int i = 1; i < rows; i++) {

                if (X[i][j] < min[j]) {
                    min[j] = X[i][j];
                }

                if (X[i][j] > max[j]) {
                    max[j] = X[i][j];
                }
            }
        }
    }

    public double[][] transform(double[][] X) {

        int rows = X.length;
        int columns = X[0].length;

        double[][] result = new double[rows][columns];

        for (int i = 0; i < rows; i++) {

            for (int j = 0; j < columns; j++) {

                if (max[j] == min[j]) {
                    result[i][j] = 0;
                } else {
                    result[i][j] = (X[i][j] - min[j])
                            / (max[j] - min[j]);
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