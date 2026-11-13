package javaml.regression;

public class DecisionTreeRegressor implements Regressor {

    private Node root;
    private int maxDepth;

    private class Node {

        int feature;
        double threshold;

        Node left;
        Node right;

        double value;
        boolean leaf;
    }

    public DecisionTreeRegressor() {
        maxDepth = 5;
    }

    public DecisionTreeRegressor(int maxDepth) {
        this.maxDepth = maxDepth;
    }

    @Override
    public void fit(double[][] X, double[] y) {

        root = buildTree(X, y, 0);
    }

    private Node buildTree(
            double[][] X,
            double[] y,
            int depth) {

        Node node = new Node();

        if (depth >= maxDepth || y.length <= 1) {

            node.leaf = true;
            node.value = mean(y);

            return node;
        }

        int bestFeature = -1;
        double bestThreshold = 0;
        double bestError = Double.MAX_VALUE;

        for (int feature = 0; feature < X[0].length; feature++) {

            for (int i = 0; i < X.length; i++) {

                double threshold = X[i][feature];

                double error = calculateSplitError(
                        X,
                        y,
                        feature,
                        threshold);

                if (error < bestError) {

                    bestError = error;
                    bestFeature = feature;
                    bestThreshold = threshold;
                }
            }
        }

        if (bestFeature == -1) {

            node.leaf = true;
            node.value = mean(y);

            return node;
        }

        int leftCount = 0;
        int rightCount = 0;

        for (int i = 0; i < X.length; i++) {

            if (X[i][bestFeature] <= bestThreshold) {
                leftCount++;
            } else {
                rightCount++;
            }
        }

        if (leftCount == 0 || rightCount == 0) {

            node.leaf = true;
            node.value = mean(y);

            return node;
        }

        double[][] leftX = new double[leftCount][X[0].length];

        double[] leftY = new double[leftCount];

        double[][] rightX = new double[rightCount][X[0].length];

        double[] rightY = new double[rightCount];

        int leftIndex = 0;
        int rightIndex = 0;

        for (int i = 0; i < X.length; i++) {

            if (X[i][bestFeature] <= bestThreshold) {

                leftX[leftIndex] = X[i];
                leftY[leftIndex] = y[i];

                leftIndex++;
            }

            else {

                rightX[rightIndex] = X[i];
                rightY[rightIndex] = y[i];

                rightIndex++;
            }
        }

        node.feature = bestFeature;
        node.threshold = bestThreshold;
        node.leaf = false;

        node.left = buildTree(
                leftX,
                leftY,
                depth + 1);

        node.right = buildTree(
                rightX,
                rightY,
                depth + 1);

        return node;
    }

    private double calculateSplitError(
            double[][] X,
            double[] y,
            int feature,
            double threshold) {

        int leftCount = 0;
        int rightCount = 0;

        double leftSum = 0;
        double rightSum = 0;

        for (int i = 0; i < X.length; i++) {

            if (X[i][feature] <= threshold) {

                leftSum += y[i];
                leftCount++;
            }

            else {

                rightSum += y[i];
                rightCount++;
            }
        }

        if (leftCount == 0 || rightCount == 0) {
            return Double.MAX_VALUE;
        }

        double leftMean = leftSum / leftCount;

        double rightMean = rightSum / rightCount;

        double error = 0;

        for (int i = 0; i < X.length; i++) {

            if (X[i][feature] <= threshold) {

                double difference = y[i] - leftMean;

                error += difference * difference;
            }

            else {

                double difference = y[i] - rightMean;

                error += difference * difference;
            }
        }

        return error / X.length;
    }

    private double mean(double[] y) {

        double sum = 0;

        for (double value : y) {
            sum += value;
        }

        return sum / y.length;
    }

    private double predictSample(
            double[] sample,
            Node node) {

        if (node.leaf) {
            return node.value;
        }

        if (sample[node.feature] <= node.threshold) {

            return predictSample(
                    sample,
                    node.left);
        }

        return predictSample(
                sample,
                node.right);
    }

    @Override
    public double[] predict(double[][] X) {

        double[] predictions = new double[X.length];

        for (int i = 0; i < X.length; i++) {

            predictions[i] = predictSample(
                    X[i],
                    root);
        }

        return predictions;
    }
}