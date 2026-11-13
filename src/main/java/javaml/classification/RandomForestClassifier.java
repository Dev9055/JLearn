package javaml.classification;

import java.util.Random;

public class RandomForestClassifier implements Classifier {

    private DecisionTreeClassifier[] trees;

    private int numberOfTrees;
    private int maxDepth;

    private Random random;

    public RandomForestClassifier() {

        numberOfTrees = 10;
        maxDepth = 5;

        random = new Random();
    }

    public RandomForestClassifier(
            int numberOfTrees,
            int maxDepth) {

        this.numberOfTrees = numberOfTrees;
        this.maxDepth = maxDepth;

        random = new Random();
    }

    @Override
    public void fit(double[][] X, double[] y) {

        trees = new DecisionTreeClassifier[numberOfTrees];

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

            trees[t] = new DecisionTreeClassifier(
                    maxDepth);

            trees[t].fit(
                    sampleX,
                    sampleY);
        }
    }

    @Override
    public int[] predict(double[][] X) {

        int[] predictions = new int[X.length];

        for (int i = 0; i < X.length; i++) {

            int zeroVotes = 0;
            int oneVotes = 0;

            double[][] sample = {
                    X[i]
            };

            for (int t = 0; t < numberOfTrees; t++) {

                int prediction = trees[t].predict(sample)[0];

                if (prediction == 0) {
                    zeroVotes++;
                } else {
                    oneVotes++;
                }
            }

            if (oneVotes > zeroVotes) {
                predictions[i] = 1;
            } else {
                predictions[i] = 0;
            }
        }

        return predictions;
    }
}