package GeneticAlgorithms;

import java.util.Random;

import static util.RandomGenerator.uniform;

public class Chromosome implements Comparable<Chromosome> {

    private int size;
    private Double score = Double.MAX_VALUE;

    public double[][] Whidden;
    public double [] bHidden;
    public double[][] Woutput;
    public double [] bOutput;
    public Random rng = new Random(123);

    public Chromosome(int size, int nInputs, int nHidden, int nOutputs) {
        this.size = size;
            this.Whidden = new double[nHidden][nInputs];
            double w_ = 1. / nInputs;

            for(int j = 0; j < nHidden; j++) {
                for(int i = 0; i < nInputs; i++) {
                    this.Whidden[j][i] = uniform(-w_, w_, rng);
                }
            }

        this.Woutput = new double[nOutputs][nHidden];
         w_ = 1. / nInputs;

        for(int j = 0; j < nOutputs; j++) {
            for(int i = 0; i < nHidden; i++) {
                this.Woutput[j][i] = uniform(-w_, w_, rng);
            }
        }

        this.bHidden = new double[nHidden];
        this.bOutput = new double[nOutputs];
    }

    public int getSize() {
        return size;
    }

    public Double getScore() {
        return score;
    }

    public void setScore(Double score) {
        this.score = score;
    }

    public void setSize(int size) {
        this.size = size;
    }

    public double[][] getWhidden() {
        return Whidden;
    }

    public void setWhidden(double[][] whidden) {
        Whidden = whidden;
    }

    public double[] getbHidden() {
        return bHidden;
    }

    public void setbHidden(double[] bHidden) {
        this.bHidden = bHidden;
    }

    public double[][] getWoutput() {
        return Woutput;
    }

    public void setWoutput(double[][] woutput) {
        Woutput = woutput;
    }

    public double[] getbOutput() {
        return bOutput;
    }

    public void setbOutput(double[] bOutput) {
        this.bOutput = bOutput;
    }

    public Random getRng() {
        return rng;
    }

    public void setRng(Random rng) {
        this.rng = rng;
    }

    @Override
    public int compareTo(Chromosome o) {
         if (Double.compare(this.getScore(), o.getScore()) < 0) {
                return -1;
         } else if (Double.compare(this.getScore(), o.getScore()) > 0) {
             return 1;
         } else {
             return  0;
         }
    }
}
