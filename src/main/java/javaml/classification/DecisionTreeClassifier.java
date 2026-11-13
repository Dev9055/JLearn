package javaml.classification;

public class DecisionTreeClassifier implements Classifier {

    private Node root;
    private int maxDepth;

    private class Node {

        int feature;
        double threshold;

        Node left;
        Node right;

        int prediction;
        boolean leaf;
    }

    public DecisionTreeClassifier() {
        maxDepth = 5;
    }

    public DecisionTreeClassifier(int maxDepth) {
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

        if (depth >= maxDepth || sameClass(y)) {

            node.leaf = true;
            node.prediction = majorityClass(y);

            return node;
        }

        int bestFeature = -1;
        double bestThreshold = 0;
        double bestGini = Double.MAX_VALUE;

        for (int feature = 0; feature < X[0].length; feature++) {

            for (int i = 0; i < X.length; i++) {

                double threshold = X[i][feature];

                double gini = calculateSplitGini(
                        X,
                        y,
                        feature,
                        threshold);

                if (gini < bestGini) {

                    bestGini = gini;
                    bestFeature = feature;
                    bestThreshold = threshold;
                }
            }
        }

        if (bestFeature == -1) {

            node.leaf = true;
            node.prediction = majorityClass(y);

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
            node.prediction = majorityClass(y);

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

    private double calculateSplitGini(
            double[][] X,
            double[] y,
            int feature,
            double threshold) {

        int leftZero = 0;
        int leftOne = 0;

        int rightZero = 0;
        int rightOne = 0;

        for (int i = 0; i < X.length; i++) {

            if (X[i][feature] <= threshold) {

                if ((int) y[i] == 0) {
                    leftZero++;
                } else {
                    leftOne++;
                }
            }

            else {

                if ((int) y[i] == 0) {
                    rightZero++;
                } else {
                    rightOne++;
                }
            }
        }

        int leftTotal = leftZero + leftOne;

        int rightTotal = rightZero + rightOne;

        if (leftTotal == 0 || rightTotal == 0) {
            return Double.MAX_VALUE;
        }

        double leftGini = gini(
                leftZero,
                leftOne);

        double rightGini = gini(
                rightZero,
                rightOne);

        int total = leftTotal + rightTotal;

        return ((double) leftTotal / total)
                * leftGini
                +
                ((double) rightTotal / total)
                        * rightGini;
    }

    private double gini(
            int zero,
            int one) {

        int total = zero + one;

        if (total == 0) {
            return 0;
        }

        double p0 = (double) zero / total;

        double p1 = (double) one / total;

        return 1
                - (p0 * p0)
                - (p1 * p1);
    }

    private boolean sameClass(double[] y) {

        for (int i = 1; i < y.length; i++) {

            if (y[i] != y[0]) {
                return false;
            }
        }

        return true;
    }

    private int majorityClass(double[] y) {

        int zero = 0;
        int one = 0;

        for (double value : y) {

            if ((int) value == 0) {
                zero++;
            } else {
                one++;
            }
        }

        if (one > zero) {
            return 1;
        }

        return 0;
    }

    private int predictSample(
            double[] sample,
            Node node) {

        if (node.leaf) {
            return node.prediction;
        }

        if (sample[node.feature] <= node.threshold) {

            return predictSample(
                    sample,
                    node.left);
        }

        else {

            return predictSample(
                    sample,
                    node.right);
        }
    }

    @Override
    public int[] predict(double[][] X) {

        int[] predictions = new int[X.length];

        for (int i = 0; i < X.length; i++) {

            predictions[i] = predictSample(
                    X[i],
                    root);
        }

        return predictions;
    }
}