package org.openjfx.gaJava.GeneticAlgorithms;

import org.openjfx.gaJava.MultiLayerNeuralNetworks.MultiLayerPerceptrons;
import org.openjfx.gaJava.mnist.ImageManipulation;
import org.openjfx.gaJava.mnist.Mnist;
import org.openjfx.gaJava.mnist.MnistMatrix;

import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

import static org.openjfx.gaJava.util.ArrayTypeConversion.convertIntToDouble;
import static org.openjfx.gaJava.util.ArrayTypeConversion.convertIntToBinaryArray;
import static org.openjfx.gaJava.util.ArrayTypeConversion.convertIntToInteger;

public class GANeuralNetwork implements Serializable {
    public Population population;
    double[][][] finalOutput;
    final Random rnd = new Random();
    int nHiddenGenes;
    int nOutputGenes;
    int nInputs;
    int nOutputs;
    public Population parents;
    public Population children;
    int size;
    double oldScore;
    int countOfSameScore = 0;
    public Chromosome bestChromose;
    public Chromosome secondBestChromosome;
    double[][] score;

    public GANeuralNetwork(int size, int nHiddenGenes, int nOutputGenes, int nInputs, int nOutputs) {
        this.population = new Population(size, nHiddenGenes, nOutputGenes, nInputs, nOutputs);
        this.nHiddenGenes = nHiddenGenes;
        this.nOutputGenes = nOutputGenes;
        this.nInputs = nInputs;
        this.nOutputs = nOutputs;
        this.parents = new Population();
        this.children = new Population();
        this.size = size;
        this.score = new double[size][1];
    }

    public void fitness(int[][] input, int[][] target, boolean print) {
        double[] doubleInput = convertIntToDouble(input);

        for (int c = 0; c < this.population.getOutputChromosomes().size(); c++) {
            Integer[] predicted = this.population.getOutputChromosomes().get(c).mlp.predict(doubleInput);

            double score = 0;
            for (int i = 0; i < predicted.length; i++) {
                score += Math.abs(target[0][i] - predicted[i]);
            }
            this.population.getOutputChromosomes().get(c).setScore(score +
                    this.population.getOutputChromosomes().get(c).getScore());
        }
    }

    public Integer[] predict(int[][] input) {
        double[] doubleInput = convertIntToDouble(input);

        return this.bestChromose.mlp.predict(doubleInput);
    }

    public void selection() {

        // select 2 best parents for breeding

        Collections.sort(this.population.getOutputChromosomes());

        ArrayList<Chromosome> outputChromosomes = new ArrayList<>();

        for (int p = 0; p < 2; p++) {
            outputChromosomes.add(this.population.getOutputChromosomes().get(p));
        }

        parents.setOutputChromosomes(outputChromosomes);
        this.setBestChromose(this.population.getOutputChromosomes().get(0));

        this.setSecondBestChromosome(this.secondBestChromosome = this.population.getOutputChromosomes().get(1));
        this.oldScore = population.getOutputChromosomes().get(0).getScore();
        this.score = new double[size][1];
    }

    public void crossover() {
        ArrayList<Chromosome> outputChromosomeChildren = new ArrayList<>();

        for (int c = 0; c < size - parents.getOutputChromosomes().size(); c++) {

            // crossover for hidden layer chromosomes
            Chromosome child = new Chromosome(size, nInputs, nHiddenGenes, nOutputs);
            int crossOverPointWeights = (int) Math.floor(Math.random() * nInputs);
            int crossOverPointBias = (int) Math.floor(Math.random() * nHiddenGenes);

            for (int i = 0; i < nHiddenGenes; i++) {
                if (crossOverPointBias < i) {
                    child.mlp.hiddenLayer.b[i] = bestChromose.mlp.hiddenLayer.b[i];
                } else {
                    child.mlp.hiddenLayer.b[i] = secondBestChromosome.mlp.hiddenLayer.b[i];
                }
                for (int j = 0; j < nInputs; j++) {
                    if (j < crossOverPointWeights) {
                        child.mlp.hiddenLayer.W[i][j] = bestChromose.mlp.hiddenLayer.W[i][j];
                    } else {
                        child.mlp.hiddenLayer.W[i][j] = secondBestChromosome.mlp.hiddenLayer.W[i][j];
                    }
                }
            }

            // crossover for output layer chromosomes
            crossOverPointWeights = (int) Math.floor(Math.random() * nHiddenGenes);
            crossOverPointBias = (int) Math.floor(Math.random() * nOutputs);

            for (int i = 0; i < nOutputs; i++) {
                if (crossOverPointBias < i) {
                    child.mlp.logisticLayer.b[i] = bestChromose.mlp.logisticLayer.b[i];
                } else {
                    child.mlp.logisticLayer.b[i] = secondBestChromosome.mlp.logisticLayer.b[i];
                }
                for (int j = 0; j < nHiddenGenes; j++) {
                    if (j < crossOverPointWeights) {
                        child.mlp.logisticLayer.W[i][j] = bestChromose.mlp.logisticLayer.W[i][j];
                    } else {
                        child.mlp.logisticLayer.W[i][j] = secondBestChromosome.mlp.logisticLayer.W[i][j];
                    }
                }
            }

            outputChromosomeChildren.add(child);
        }

        this.children.setOutputChromosomes(outputChromosomeChildren);
    }

    public void mutation() {
        int numberOfMutationsHidden;
        for (int c = 0; c < this.children.getOutputChromosomes().size(); c++) {
            // W 10x784
            numberOfMutationsHidden = (int) Math.abs(Math.random() * nInputs * 0.05);

            for (int i = 0; i < numberOfMutationsHidden; i++) {
                int m = (int) Math.floor(Math.random() *
                        this.children.getOutputChromosomes().get(c).mlp.hiddenLayer.W.length);
                int mutationPoint = (int) Math.floor(Math.random() *
                        this.children.getOutputChromosomes().get(c).mlp.hiddenLayer.W[0].length);
                this.children.getOutputChromosomes().get(c).mlp.hiddenLayer.W[m][mutationPoint]
                        += rnd.nextGaussian() * 0.1;
                this.children.getOutputChromosomes().get(c).mlp.hiddenLayer.b[mutationPoint % nOutputs]
                        += rnd.nextGaussian() * 0.1;
            }

            int numberOfMutationsOutput;
            // W 10x10
            for (int i = 0; i < nOutputs; i++) {
                numberOfMutationsOutput = (int) (Math.random() * nOutputs * 0.2);
                for (int j = 0; j < numberOfMutationsOutput; j++) {
                    int mutationPoint = (int) Math.floor(Math.abs(Math.random() *
                            this.children.getOutputChromosomes().get(c).mlp.logisticLayer.W[0].length));
                    this.children.getOutputChromosomes().get(c).mlp.logisticLayer.W[i][mutationPoint]
                            += rnd.nextGaussian() * 0.1;
                    this.children.getOutputChromosomes().get(c).mlp.logisticLayer.b[mutationPoint % nOutputs]
                            += rnd.nextGaussian() * 0.1;
                }
            }
        }

        this.population.setOutputChromosomes(this.parents.getOutputChromosomes());

        for (int c = 0; c < this.children.getOutputChromosomes().size(); c++) {
            this.population.getOutputChromosomes().add(this.children.getOutputChromosomes().get(c));
        }
    }

    public Chromosome getBestChromose() {
        return bestChromose;
    }

    public void setBestChromose(Chromosome bestChromose) {
        this.bestChromose = bestChromose;
    }

    public Chromosome getSecondBestChromosome() {
        return secondBestChromosome;
    }

    public void setSecondBestChromosome(Chromosome secondBestChromosome) {
        this.secondBestChromosome = secondBestChromosome;
    }

    public Population getPopulation() {
        return population;
    }

    public void setPopulation(Population population) {
        this.population = population;
    }

    public static GANeuralNetwork getGANetwork() throws Exception {
        //MnistMatrix[] inputMatrix = Mnist.getMnistTrainMatrix();

        GANeuralNetwork ga = new GANeuralNetwork(180, 10,
                3, 784, 10);

        String gaFileName = "NeuralNetworkSerialized_GA" + ".ser";
        //String gaFileName = "NeuralNetworkSerialized_Mlp" + ".ser";

        File gaFile = new File(gaFileName);

        if (gaFile.exists()) {
            System.out.println("file exists");
            FileInputStream fileIn = new FileInputStream(gaFileName);

            if (fileIn != null) {
                ObjectInputStream in = new ObjectInputStream(fileIn);
                double[][] hiddenWeightsMatrix = (double[][]) in.readObject();
                double[] biasHidden = (double[]) in.readObject();
                double[][] outputWeightsMatrix = (double[][]) in.readObject();
                double[] biasOutput = (double[]) in.readObject();
                ga.bestChromose = ga.population.getOutputChromosomes().get(0);
                ga.bestChromose.mlp.hiddenLayer.W = hiddenWeightsMatrix;
                ga.bestChromose.mlp.hiddenLayer.b = biasHidden;
                ga.bestChromose.mlp.logisticLayer.W = outputWeightsMatrix;
                ga.bestChromose.mlp.logisticLayer.b = biasOutput;

                ga.population.getOutputChromosomes().get(0).mlp.hiddenLayer.W = hiddenWeightsMatrix;
                ga.population.getOutputChromosomes().get(0).mlp.hiddenLayer.b = biasHidden;
                ga.population.getOutputChromosomes().get(0).mlp.logisticLayer.W = outputWeightsMatrix;
                ga.population.getOutputChromosomes().get(0).mlp.logisticLayer.b = biasOutput;

                in.close();
                fileIn.close();
            }
        } else {
            System.out.println("GA file does not exist");
        }
        return ga;
    }

    public static void saveGANetwork(GANeuralNetwork ga) throws Exception {
        String gaFileName = "NeuralNetworkSerialized_GA" + ".ser";
        FileOutputStream fileOut = new FileOutputStream(gaFileName);
        ObjectOutputStream out = new ObjectOutputStream(fileOut);
        out.writeObject(ga.bestChromose.mlp.hiddenLayer.W);
        out.writeObject(ga.bestChromose.mlp.hiddenLayer.b);
        out.writeObject(ga.bestChromose.mlp.logisticLayer.W);
        out.writeObject(ga.bestChromose.mlp.logisticLayer.b);
        out.flush();
        out.close();
        fileOut.close();
        System.out.println("Serialized data is saved in NeuralNetworkSerializednetwork.ser");
    }

    public static void saveMlpNetwork(MultiLayerPerceptrons mlp) throws Exception {
        String mlpFileName = "NeuralNetworkSerialized_Mlp" + ".ser";
        FileOutputStream fileOut = new FileOutputStream(mlpFileName);
        ObjectOutputStream out = new ObjectOutputStream(fileOut);
        out.writeObject(mlp.hiddenLayer.W);
        out.writeObject(mlp.hiddenLayer.b);
        out.writeObject(mlp.logisticLayer.W);
        out.writeObject(mlp.logisticLayer.b);
        out.flush();
        out.close();
        fileOut.close();
        System.out.println("Serialized data is saved in " + mlpFileName);
    }

    public static MultiLayerPerceptrons getMlpNetwork() throws Exception {
        //  MnistMatrix[] inputMatrix = Mnist.getMnistTrainMatrix();

        int nInputs = 784;
        MultiLayerPerceptrons mlp = new MultiLayerPerceptrons(nInputs, 10, 10, new Random(123));

        String mlpFileName = "NeuralNetworkSerialized_Mlp" + ".ser";

        File gaFile = new File(mlpFileName);

        if (gaFile.exists()) {
            System.out.println("file exists");
            FileInputStream fileIn = new FileInputStream(mlpFileName);

            if (fileIn != null) {
                ObjectInputStream in = new ObjectInputStream(fileIn);
                double[][] hiddenWeightsMatrix = (double[][]) in.readObject();
                double[] biasHidden = (double[]) in.readObject();
                double[][] outputWeightsMatrix = (double[][]) in.readObject();
                double[] biasOutput = (double[]) in.readObject();
                mlp.hiddenLayer.W = hiddenWeightsMatrix;
                mlp.hiddenLayer.b = biasHidden;
                mlp.logisticLayer.W = outputWeightsMatrix;
                mlp.logisticLayer.b = biasOutput;

                in.close();
                fileIn.close();
            }
        } else {
            System.out.println("MLP file does not exist");
        }
        return mlp;
    }

    public static void train() throws Exception {
        MnistMatrix[] inputMatrix = Mnist.getMnistTrainMatrix();
        GANeuralNetwork ga = getGANetwork();
        int nInputs = inputMatrix[0].getnCols() * inputMatrix[0].getnRows();
        MultiLayerPerceptrons mlp = getMlpNetwork();

        double[][] trainDataForMlp = new double[inputMatrix.length][nInputs];
        int[][] targetForMlp = new int[inputMatrix.length][10];

        for (int in = 0; in < inputMatrix.length; in++) {
            int[][] testInput = inputMatrix[in].getData();
            int[] label = convertIntToBinaryArray(inputMatrix[in].getLabel());
            trainDataForMlp[in] = convertIntToDouble(testInput);
            targetForMlp[in] = label;
        }

        int iterations = 200;

        loop:
        for (int i = 0; i < iterations; i++) {
            double score = 0.0;

            for (int c = 0; c < ga.population.getOutputChromosomes().size(); c++) {
                ga.population.getOutputChromosomes().get(c).setScore(0.00);
            }

            for (int in = 0; in < inputMatrix.length; in++) {
                int[][] testInput = inputMatrix[in].getData();
                int[] label = convertIntToBinaryArray(inputMatrix[in].getLabel());
                int[][] testOutput = new int[][]{label};
                // System.out.print(testOutput[0][0] + ", " + testOutput[0][1] + ", " + testOutput[0][2] + "\n");
                ga.fitness(testInput, testOutput, false);
            }

            ga.selection();
            ga.crossover();
            ga.mutation();

            //mlp.train(trainDataForMlp, targetForMlp, inputMatrix.length, 0.01);

            //System.out.println("Score: " + ga.parents.getOutputChromosomes().get(0).getScore());

            score += ga.bestChromose.getScore();

            if (i % 5 == 0) {
                System.out.println(score);
            }
            if (i % 20 == 0) {
                saveGANetwork(ga);
                saveMlpNetwork(mlp);
                System.out.println("processed : " + ((double) i / (double) iterations) * 100 + "%");
            }
            if (i == iterations - 1) {
                saveGANetwork(ga);
                saveMlpNetwork(mlp);
                System.out.println(score);
                System.out.println("test _____________________________");
                int count = 0;

                for (int in = 0; in < inputMatrix.length; in++) {
                    int[][] testInput = inputMatrix[in].getData();
                    int[] label = convertIntToBinaryArray(inputMatrix[in].getLabel());
                    int[][] testOutput = new int[][]{label};

                    if (in < 500) {
                        ImageManipulation.saveIntToImage(testInput, "images/" + in + ".png");
                    }
                    Integer[] predicted = ga.predict(testInput);

                    int localCount = 0;
                    for (int j = 0; j < predicted.length; j++) {
                        if (testOutput[0][j] == predicted[j]) {
                            localCount++;
                        }
                    }
                    if (localCount == predicted.length) {
                        count++;
                    }
                }
                System.out.println("recognized with genetic algorithm " + count);
                System.out.println("total dataset  " + inputMatrix.length);

                System.out.println("++++++++++++++++++++++++++++++++++++++++++++++++");
                System.out.println("predicted with neural network: ");

                count = 0;
                for (int in = 0; in < inputMatrix.length; in++) {
                    int[][] testInput = inputMatrix[in].getData();
                    int[] label = convertIntToBinaryArray(inputMatrix[in].getLabel());
                    int[][] testOutput = new int[][]{label};

                    if (in < 30) {
                        ImageManipulation.saveIntToImage(testInput, "images/" + in + ".png");
                    }

                    double[] testInputMlp = convertIntToDouble(testInput);
                    Integer[] predictedMlp = mlp.predict(testInputMlp);
                    Integer[] targetOutput = convertIntToInteger(label);

//                    System.out.println("++++++++++++++++++++++++++++");
//                    System.out.print("predicted with neural network: ");
//                    System.out.println();
//                    System.out.print("predicted     " + Arrays.asList(predictedMlp).indexOf(1) + ", ");
//                    System.out.println("target     " + Arrays.asList(targetOutput).indexOf(1));
                    int localCount = 0;
                    for (int j = 0; j < predictedMlp.length; j++) {
                        if (testOutput[0][j] == predictedMlp[j]) {
                            localCount++;
                        }
                    }
                    if (localCount == predictedMlp.length) {
                        count++;
                    }
                    //System.out.println("++++++++++++++++++++++++++++");
                }

                System.out.println("recognized with neural network: " + count);
                System.out.println("total dataset size  " + inputMatrix.length);

                break loop;
            }
        }
    }

    public static void main(String... args) throws Exception {
        GANeuralNetwork.train();
    }
}
