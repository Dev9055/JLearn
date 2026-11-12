package javaml.data;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

public class CSVReader {

    public static Dataset read(String filePath) {

        ArrayList<double[]> rows = readFile(filePath);

        if (rows == null || rows.isEmpty()) {
            return null;
        }

        int numberOfRows = rows.size();
        int numberOfColumns = rows.get(0).length;

        double[][] X = new double[numberOfRows][numberOfColumns - 1];

        double[] y = new double[numberOfRows];

        for (int i = 0; i < numberOfRows; i++) {

            double[] row = rows.get(i);

            for (int j = 0; j < numberOfColumns - 1; j++) {
                X[i][j] = row[j];
            }

            y[i] = row[numberOfColumns - 1];
        }

        return new Dataset(X, y);
    }

    public static double[][] readFeatures(String filePath) {

        ArrayList<double[]> rows = readFile(filePath);

        if (rows == null || rows.isEmpty()) {
            return null;
        }

        int numberOfRows = rows.size();
        int numberOfColumns = rows.get(0).length;

        double[][] X = new double[numberOfRows][numberOfColumns];

        for (int i = 0; i < numberOfRows; i++) {

            for (int j = 0; j < numberOfColumns; j++) {

                X[i][j] = rows.get(i)[j];
            }
        }

        return X;
    }

    private static ArrayList<double[]> readFile(String filePath) {

        ArrayList<double[]> rows = new ArrayList<>();

        try {

            BufferedReader br = new BufferedReader(
                    new FileReader(filePath));

            String line;

            // Skip header
            br.readLine();

            while ((line = br.readLine()) != null) {

                String[] values = line.split(",");

                double[] row = new double[values.length];

                for (int i = 0; i < values.length; i++) {

                    row[i] = Double.parseDouble(
                            values[i].trim());
                }

                rows.add(row);
            }

            br.close();

        } catch (IOException e) {

            System.out.println(
                    "Error reading CSV file.");

            return null;
        }

        return rows;
    }
}