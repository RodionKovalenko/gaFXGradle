package GeneticAlgorithms;

import java.util.ArrayList;

public class Population {
    ArrayList<Chromosome> hiddenChromosomes;
    ArrayList<Chromosome> outputChromosomes;


    public Population() {

    }

    public Population(int size, int hiddenGenesSize, int outputGenesSize, int nInputs,int nOutputs) {
        this.hiddenChromosomes = new ArrayList<>();
        this.outputChromosomes = new ArrayList<>();

        for (int i = 0; i < size; i++) {
            Chromosome outputChromosome = new Chromosome(size, nInputs, hiddenGenesSize, nOutputs);
            this.outputChromosomes.add(outputChromosome);
        }
    }

    public ArrayList<Chromosome> getOutputChromosomes() {
        return outputChromosomes;
    }

    public void setOutputChromosomes(ArrayList<Chromosome> outputChromosomes) {
        this.outputChromosomes = outputChromosomes;
    }
}
