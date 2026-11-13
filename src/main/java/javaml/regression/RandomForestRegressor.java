package javaml.regression;

import java.util.Random;

public class RandomForestRegressor implements Regressor {

    private DecisionTreeRegressor[] trees;

    private int numberOfTrees;
    private int maxDepth;

    private Random random;

    public RandomForestRegressor() {

        numberOfTrees = 10;
        maxDepth = 5;

        random = new Random();
    }

    public RandomForestRegressor(
            int numberOfTrees,
            int maxDepth) {

        this.numberOfTrees = numberOfTrees;
        this.maxDepth = maxDepth;

        random = new Random();
    }

    @Override
    public void fit(double[][] X, double[] y) {

        trees = new DecisionTreeRegressor[numberOfTrees];

        for (int t = 0; t < numberOfTrees; t++) {

            double[][] sampleX = new double[X.length][X[0].length];

            double[] sampleY = new double[y.length];

            for (int i = 0; i < X.length; i++) {

                int randomIndex = random.nextInt(X.length);

                for (int j = 0; j < X[0].length; j++) {

                    sampleX[i][j] = X[randomIndex][j];
                }

                sampleY[i] = y[randomIndex];
            }

            trees[t] = new DecisionTreeRegressor(
                    maxDepth);

            trees[t].fit(
                    sampleX,
                    sampleY);
        }
    }

    @Override
    public double[] predict(double[][] X) {

        double[] predictions = new double[X.length];

        for (int t = 0; t < numberOfTrees; t++) {

            double[] treePredictions = trees[t].predict(X);

            for (int i = 0; i < X.length; i++) {

                predictions[i] += treePredictions[i];
            }
        }

        for (int i = 0; i < X.length; i++) {

            predictions[i] = predictions[i]
                    / numberOfTrees;
        }

        return predictions;
    }
}