package org.openjfx.gaJava.mnist;

import java.io.File;
import java.io.IOException;

public class Main {

    public static void main(String[] args) throws Exception {

        Mnist mnist = new Mnist();
        MnistMatrix[] trainData = mnist.getMnistTrainMatrix();

        for (int i = 0; i < trainData.length; i++) {
            ImageManipulation.saveIntToImage(trainData[i].getData(), "images/" + i + ".png");
        }
    }
}
