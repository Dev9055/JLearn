package javaml;

import javaml.data.CSVReader;
import javaml.data.Dataset;

import javaml.preprocessing.TrainTestSplit;
import javaml.preprocessing.StandardScaler;

import javaml.classification.*;

import javaml.metrics.ClassificationMetrics;
import javaml.metrics.ConfusionMatrix;

public class ClassificationTest {

    public static void main(String[] args) {

        Dataset data = CSVReader.read("datasets/student_pass.csv");

        TrainTestSplit split = new TrainTestSplit(
                data.getX(),
                data.getY(),
                0.2);

        StandardScaler scaler = new StandardScaler();

        scaler.fit(split.getXTrain());

        double[][] XTrain = scaler.transform(split.getXTrain());

        double[][] XTest = scaler.transform(split.getXTest());

        double[] yTrain = split.getYTrain();

        double[] yTest = split.getYTest();

        Classifier[] models = {

                new LogisticRegression(),

                new Perceptron(),

                new KNNClassifier(3),

                new GaussianNaiveBayes(),

                new DecisionTreeClassifier(5),

                new RandomForestClassifier(20, 5),

                new SVM()
        };

        String[] names = {

                "Logistic Regression",
                "Perceptron",
                "KNN Classifier",
                "Gaussian Naive Bayes",
                "Decision Tree",
                "Random Forest",
                "SVM"
        };

        for (int i = 0; i < models.length; i++) {

            models[i].fit(
                    XTrain,
                    yTrain);

            int[] predictions = models[i].predict(XTest);

            double accuracy = ClassificationMetrics.accuracy(
                    yTest,
                    predictions);

            double precision = ClassificationMetrics.precision(
                    yTest,
                    predictions);

            double recall = ClassificationMetrics.recall(
                    yTest,
                    predictions);

            double f1 = ClassificationMetrics.f1Score(
                    yTest,
                    predictions);

            System.out.println();
            System.out.println(
                    "===== " + names[i] + " =====");

            System.out.println(
                    "Accuracy: " + accuracy);

            System.out.println(
                    "Precision: " + precision);

            System.out.println(
                    "Recall: " + recall);

            System.out.println(
                    "F1 Score: " + f1);

            ConfusionMatrix matrix = new ConfusionMatrix(
                    yTest,
                    predictions);

            matrix.print();
        }
    }
}