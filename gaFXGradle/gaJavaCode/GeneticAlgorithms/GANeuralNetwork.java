package GeneticAlgorithms;

import MultiLayerNeuralNetworks.MultiLayerPerceptrons;
import mnist.ImageManipulation;
import mnist.Mnist;
import mnist.MnistMatrix;

import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

import static util.ArrayTypeConversion.*;

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

    public MultiLayerPerceptrons mlp;

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
        this.mlp = new MultiLayerPerceptrons(nInputs, nHiddenGenes, nOutputs, new Random(new Random().nextInt()));
    }

    public void fitness(int[][] input, int[][] target, boolean print) {
        double[] doubleInput = convertIntToDouble(input);

        for (int c = 0; c < this.population.getOutputChromosomes().size(); c++) {
            this.setWeightsInMlp(c);
            Integer[] predicted = this.mlp.predict(doubleInput);

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
        this.setWeightsInMlp(0);
        return this.mlp.predict(doubleInput);
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
                    child.bHidden[i] = bestChromose.bHidden[i];
                } else {
                    child.bHidden[i] = secondBestChromosome.bHidden[i];
                }
                for (int j = 0; j < nInputs; j++) {
                    if (j < crossOverPointWeights) {
                        child.Whidden[i][j] = bestChromose.Whidden[i][j];
                    } else {
                        child.Whidden[i][j] = secondBestChromosome.Whidden[i][j];
                    }
                }
            }

            // crossover for output layer chromosomes
            crossOverPointWeights = (int) Math.floor(Math.random() * nHiddenGenes);
            crossOverPointBias = (int) Math.floor(Math.random() * nOutputs);

            for (int i = 0; i < nOutputs; i++) {
                if (crossOverPointBias < i) {
                    child.bOutput[i] = bestChromose.bOutput[i];
                } else {
                    child.bOutput[i] = secondBestChromosome.bOutput[i];
                }
                for (int j = 0; j < nHiddenGenes; j++) {
                    if (j < crossOverPointWeights) {
                        child.Woutput[i][j] = bestChromose.Woutput[i][j];
                    } else {
                        child.Woutput[i][j] = secondBestChromosome.Woutput[i][j];
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
                        this.children.getOutputChromosomes().get(c).Whidden.length);
                int mutationPoint = (int) Math.floor(Math.random() *
                        this.children.getOutputChromosomes().get(c).Whidden[0].length);
                this.children.getOutputChromosomes().get(c).Whidden[m][mutationPoint]
                        += rnd.nextGaussian() * 0.5;
                this.children.getOutputChromosomes().get(c).bHidden[mutationPoint % nOutputs]
                        += rnd.nextGaussian() * 0.5;
            }

            int numberOfMutationsOutput;
            // W 10x10
            for (int i = 0; i < nOutputs; i++) {
                numberOfMutationsOutput = (int) (Math.random() * nOutputs * 0.3);

                for (int j = 0; j < numberOfMutationsOutput; j++) {
                    int mutationPoint = (int) Math.floor(Math.abs(Math.random() *
                            this.children.getOutputChromosomes().get(c).Woutput[0].length));
                    this.children.getOutputChromosomes().get(c).Woutput[i][mutationPoint]
                            += rnd.nextGaussian() * 0.5;
                    this.children.getOutputChromosomes().get(c).bOutput[mutationPoint % nOutputs]
                            += rnd.nextGaussian() * 0.5;
                }
            }
        }

        this.population.setOutputChromosomes(this.parents.getOutputChromosomes());

        for (int c = 0; c < this.children.getOutputChromosomes().size(); c++) {
            this.population.getOutputChromosomes().add(this.children.getOutputChromosomes().get(c));
        }
    }

    public void setWeightsInMlp(int chromosomeIndex) {
        double [][] Whidden = this.population.getOutputChromosomes().get(chromosomeIndex).getWhidden();
        double [] bHidden = this.population.getOutputChromosomes().get(chromosomeIndex).getbHidden();
        double [][] WOutput = this.population.getOutputChromosomes().get(chromosomeIndex).getWoutput();
        double [] bOutput = this.population.getOutputChromosomes().get(chromosomeIndex).getbOutput();
        this.mlp.hiddenLayer.W = Whidden;
        this.mlp.hiddenLayer.b = bHidden;
        this.mlp.logisticLayer.W = WOutput;
        this.mlp.logisticLayer.b = bOutput;
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
                ga.bestChromose.Whidden = hiddenWeightsMatrix;
                ga.bestChromose.bHidden = biasHidden;
                ga.bestChromose.Woutput = outputWeightsMatrix;
                ga.bestChromose.bOutput = biasOutput;

                ga.population.getOutputChromosomes().get(0).Whidden = hiddenWeightsMatrix;
                ga.population.getOutputChromosomes().get(0).bHidden = biasHidden;
                ga.population.getOutputChromosomes().get(0).Woutput = outputWeightsMatrix;
                ga.population.getOutputChromosomes().get(0).bOutput = biasOutput;

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
        out.writeObject(ga.bestChromose.Whidden);
        out.writeObject(ga.bestChromose.bHidden);
        out.writeObject(ga.bestChromose.Woutput);
        out.writeObject(ga.bestChromose.bOutput);
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
        MultiLayerPerceptrons mlpSaved = new MultiLayerPerceptrons(nInputs, 10, 10, new Random(123));

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
                mlpSaved.hiddenLayer.W = hiddenWeightsMatrix;
                mlpSaved.hiddenLayer.b = biasHidden;
                mlpSaved.logisticLayer.W = outputWeightsMatrix;
                mlpSaved.logisticLayer.b = biasOutput;

                in.close();
                fileIn.close();
            }
        } else {
            System.out.println("MLP file does not exist");
        }
        return mlpSaved;
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

            mlp.train(trainDataForMlp, targetForMlp, inputMatrix.length, 0.01);

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
