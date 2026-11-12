package javaml.preprocessing;

import java.util.Random;

public class TrainTestSplit {

    private double[][] XTrain;
    private double[][] XTest;

    private double[] yTrain;
    private double[] yTest;

    public TrainTestSplit(
            double[][] X,
            double[] y,
            double testSize) {

        // Validation
        if (X == null || y == null) {
            throw new IllegalArgumentException(
                    "X and y cannot be null.");
        }

        if (X.length == 0) {
            throw new IllegalArgumentException(
                    "Dataset cannot be empty.");
        }

        if (X.length != y.length) {
            throw new IllegalArgumentException(
                    "X and y must have the same number of rows.");
        }

        if (X.length < 2) {
            throw new IllegalArgumentException(
                    "Dataset must contain at least 2 rows.");
        }

        if (testSize <= 0 || testSize >= 1) {
            throw new IllegalArgumentException(
                    "Test size must be between 0 and 1.");
        }

        int rows = X.length;
        int features = X[0].length;

        int testRows = (int) (rows * testSize);

        // Make sure at least one row goes to testing
        if (testRows == 0) {
            testRows = 1;
        }

        int trainRows = rows - testRows;

        // Create indexes
        int[] indexes = new int[rows];

        for (int i = 0; i < rows; i++) {
            indexes[i] = i;
        }

        // Shuffle indexes
        Random random = new Random();

        for (int i = rows - 1; i > 0; i--) {

            int j = random.nextInt(i + 1);

            int temp = indexes[i];

            indexes[i] = indexes[j];

            indexes[j] = temp;
        }

        // Create train and test arrays
        XTrain = new double[trainRows][features];

        XTest = new double[testRows][features];

        yTrain = new double[trainRows];

        yTest = new double[testRows];

        // Training data
        for (int i = 0; i < trainRows; i++) {

            int row = indexes[i];

            for (int j = 0; j < features; j++) {

                XTrain[i][j] = X[row][j];
            }

            yTrain[i] = y[row];
        }

        // Testing data
        for (int i = 0; i < testRows; i++) {

            int row = indexes[trainRows + i];

            for (int j = 0; j < features; j++) {

                XTest[i][j] = X[row][j];
            }

            yTest[i] = y[row];
        }
    }

    public double[][] getXTrain() {
        return XTrain;
    }

    public double[][] getXTest() {
        return XTest;
    }

    public double[] getYTrain() {
        return yTrain;
    }

    public double[] getYTest() {
        return yTest;
    }
}