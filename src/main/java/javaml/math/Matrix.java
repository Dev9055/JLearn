package javaml.math;

public class Matrix {

    private double[][] data;
    private int rows;
    private int columns;

    public Matrix(double[][] data) {
        rows = data.length;
        columns = data[0].length;

        this.data = new double[rows][columns];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                this.data[i][j] = data[i][j];
            }
        }
    }

    public Matrix(int rows, int columns) {
        this.rows = rows;
        this.columns = columns;

        data = new double[rows][columns];
    }

    public int getRows() {
        return rows;
    }

    public int getColumns() {
        return columns;
    }

    public double get(int row, int column) {
        return data[row][column];
    }

    public void set(int row, int column, double value) {
        data[row][column] = value;
    }

    public Matrix add(Matrix other) {
        Matrix result = new Matrix(rows, columns);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                result.data[i][j] = data[i][j] + other.data[i][j];
            }
        }

        return result;
    }

    public Matrix subtract(Matrix other) {
        Matrix result = new Matrix(rows, columns);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                result.data[i][j] = data[i][j] - other.data[i][j];
            }
        }

        return result;
    }

    public Matrix multiply(double value) {
        Matrix result = new Matrix(rows, columns);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                result.data[i][j] = data[i][j] * value;
            }
        }

        return result;
    }

    public Matrix multiply(Matrix other) {
        Matrix result = new Matrix(rows, other.columns);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < other.columns; j++) {
                for (int k = 0; k < columns; k++) {

                    result.data[i][j] += data[i][k] * other.data[k][j];
                }
            }
        }

        return result;
    }

    public Matrix transpose() {
        Matrix result = new Matrix(columns, rows);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {

                result.data[j][i] = data[i][j];
            }
        }

        return result;
    }

    public void print() {
        for (int i = 0; i < rows; i++) {

            for (int j = 0; j < columns; j++) {
                System.out.print(data[i][j] + " ");
            }

            System.out.println();
        }
    }
}