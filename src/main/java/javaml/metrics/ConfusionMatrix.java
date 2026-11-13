package javaml.metrics;

public class ConfusionMatrix {

    private int truePositive;
    private int trueNegative;
    private int falsePositive;
    private int falseNegative;

    public ConfusionMatrix(
            double[] actual,
            int[] predicted) {

        calculate(actual, predicted);
    }

    private void calculate(
            double[] actual,
            int[] predicted) {

        for (int i = 0; i < actual.length; i++) {

            int actualValue = (int) actual[i];

            int predictedValue = predicted[i];

            if (actualValue == 1
                    && predictedValue == 1) {

                truePositive++;
            }

            else if (actualValue == 0
                    && predictedValue == 0) {

                trueNegative++;
            }

            else if (actualValue == 0
                    && predictedValue == 1) {

                falsePositive++;
            }

            else if (actualValue == 1
                    && predictedValue == 0) {

                falseNegative++;
            }
        }
    }

    public int getTruePositive() {
        return truePositive;
    }

    public int getTrueNegative() {
        return trueNegative;
    }

    public int getFalsePositive() {
        return falsePositive;
    }

    public int getFalseNegative() {
        return falseNegative;
    }

    public void print() {

        System.out.println("Confusion Matrix:");

        System.out.println();

        System.out.println(
                "        Predicted");

        System.out.println(
                "        0     1");

        System.out.println(
                "Actual 0  "
                        + trueNegative
                        + "     "
                        + falsePositive);

        System.out.println(
                "       1  "
                        + falseNegative
                        + "     "
                        + truePositive);
    }
}