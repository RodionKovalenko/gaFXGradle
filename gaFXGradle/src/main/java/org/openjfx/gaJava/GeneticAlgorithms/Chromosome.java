package org.openjfx.gaJava.GeneticAlgorithms;

import org.openjfx.gaJava.MultiLayerNeuralNetworks.MultiLayerPerceptrons;

import java.util.Random;

public class Chromosome implements Comparable<Chromosome> {

    private int size;
    private Double score = Double.MAX_VALUE;

    public MultiLayerPerceptrons mlp;

    public Chromosome(int size, int nInputs, int nHidden, int nOutputs) {
        this.size = size;
        this.mlp = new MultiLayerPerceptrons(nInputs, nHidden, nOutputs, new Random(new Random().nextInt()));
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
