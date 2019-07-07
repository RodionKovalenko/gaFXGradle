package util;

public class ArrayTypeConversion {

    public static double[] convertIntToDouble(int[][] data) {

        double[] convertedArray = new double[data.length * data[0].length];

        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[0].length; j++) {
                convertedArray[i * j] = data[i][j];
            }
        }

        return convertedArray;
    }

    public static int[] convertIntToBinaryArray(int value) {

        int[] binaryValues = new int[10];

        for (int i = 0; i < binaryValues.length; i++) {
            if (i == value) {
                binaryValues[i] = 1;
            } else {
                binaryValues[i] = 0;
            }
        }

        return binaryValues;
    }

    public static Integer[] convertIntToInteger(int [] values) {

        Integer [] integerValues = new Integer[values.length];

        for (int i = 0; i < integerValues.length; i++) {
            integerValues[i] = Integer.valueOf(values[i]);
        }
        return integerValues;
    }
}
