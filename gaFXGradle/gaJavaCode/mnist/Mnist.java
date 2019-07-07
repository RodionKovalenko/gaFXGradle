package mnist;


public class Mnist {
    public static String workingDir = System.getProperty("user.dir");
    public static String projectPath = "";

    public static int[][][] getMnistTrainData() throws Exception {
        MnistMatrix[] trainMatrix = new MnistDataReader().readData(
                workingDir + projectPath + "\\mnist\\data\\train-images.idx3-ubyte",
                workingDir + projectPath +"\\mnist\\data\\train-labels.idx1-ubyte");

        int[][][] mnistTrainData = new int[trainMatrix.length]
                [trainMatrix[0].getData().length]
                [trainMatrix[0].getData()[0].length];

        for (int i = 0; i < trainMatrix.length; i++) {
            mnistTrainData[i] = trainMatrix[i].getData();
        }

        return mnistTrainData;
    }

    public static int[][][] getMnistTestData() throws Exception {
        MnistMatrix[] testMatrix = new MnistDataReader().readData(
                workingDir + projectPath + "\\mnist\\data\\t10k-images.idx3-ubyte",
                workingDir + projectPath + "\\mnist\\data\\t10k-labels.idx1-ubyte");

        int[][][] mnistTestData = new int[testMatrix.length]
                [testMatrix[0].getData().length]
                [testMatrix[0].getData()[0].length];

        for (int i = 0; i < testMatrix.length; i++) {
            mnistTestData[i] = testMatrix[i].getData();
        }

        return mnistTestData;
    }

    public static MnistMatrix[] getMnistTrainMatrix() throws Exception {
        MnistMatrix[] trainMatrix = new MnistDataReader().readData(
                workingDir + projectPath + "\\mnist\\data\\train-images.idx3-ubyte",
                workingDir + projectPath + "\\mnist\\data\\train-labels.idx1-ubyte");
        return trainMatrix;
    }

    public static MnistMatrix[] getMnistTestMatrix() throws Exception {
        MnistMatrix[] testMatrix = new MnistDataReader().readData(
                workingDir + projectPath + "\\mnist\\data\\t10k-images.idx3-ubyte",
                workingDir + projectPath + "\\mnist\\data\\t10k-labels.idx1-ubyte");
        return testMatrix;
    }

    public static void printMnistMatrix(final MnistMatrix matrix) {
        System.out.println("label: " + matrix.getLabel());
        for (int r = 0; r < matrix.getNumberOfRows(); r++) {
            for (int c = 0; c < matrix.getNumberOfColumns(); c++) {
                System.out.print(matrix.getValue(r, c) + " ");
            }
            System.out.println();
        }
    }
}
