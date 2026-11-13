package javaml;

import javaml.data.CSVReader;
import javaml.data.Dataset;

import javaml.preprocessing.TrainTestSplit;
import javaml.preprocessing.StandardScaler;

import javaml.regression.*;

import javaml.metrics.RegressionMetrics;

public class RegressionTest {

    public static void main(String[] args) {

        Dataset data = CSVReader.read("datasets/students.csv");

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

        Regressor[] models = {

                new LinearRegression(),

                new PolynomialRegression(2),

                new RidgeRegression(),

                new LassoRegression(),

                new KNNRegressor(3),

                new DecisionTreeRegressor(5),

                new RandomForestRegressor(20, 5),

                new SVR()
        };

        String[] names = {

                "Linear Regression",
                "Polynomial Regression",
                "Ridge Regression",
                "Lasso Regression",
                "KNN Regression",
                "Decision Tree",
                "Random Forest",
                "SVR"
        };

        for (int i = 0; i < models.length; i++) {

            models[i].fit(
                    XTrain,
                    yTrain);

            double[] predictions = models[i].predict(XTest);

            double mae = RegressionMetrics.mae(
                    yTest,
                    predictions);

            double rmse = RegressionMetrics.rmse(
                    yTest,
                    predictions);

            double r2 = RegressionMetrics.r2Score(
                    yTest,
                    predictions);

            System.out.println(
                    "\n" + names[i]);

            System.out.println(
                    "MAE: " + mae);

            System.out.println(
                    "RMSE: " + rmse);

            System.out.println(
                    "R2: " + r2);
        }
    }
}