import javaml.regression.LinearRegression;
import javaml.regression.Regressor;
import javaml.metrics.RegressionMetrics;

public class RegressionExample {

    public static void main(String[] args) {

        double[][] X = {
                { 1 },
                { 2 },
                { 3 },
                { 4 },
                { 5 }
        };

        double[] y = {
                2, 4, 6, 8, 10
        };

        Regressor model = new LinearRegression(
                0.01,
                5000);

        model.fit(X, y);

        double[] predictions = model.predict(X);

        System.out.println(
                "RMSE: "
                        + RegressionMetrics.rmse(
                                y,
                                predictions));

        System.out.println(
                "R2: "
                        + RegressionMetrics.r2Score(
                                y,
                                predictions));
    }
}