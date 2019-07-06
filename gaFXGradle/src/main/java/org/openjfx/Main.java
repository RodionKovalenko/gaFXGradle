package org.openjfx;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.stage.Stage;

import java.io.File;
import java.net.URI;
import java.net.URL;

public class Main extends Application {
    public static Stage applicationStage;

    @Override
    public void start(Stage stage) throws Exception {


        String gaFileName = "NeuralNetworkSerialized_GA.ser";

        File file=new File(gaFileName);

        System.out.println("gaFile exists " + file.exists());
        System.out.println(System.getProperty("user.dir"));
        System.out.println(file.getAbsolutePath());
        applicationStage = stage;
        Parent root = FXMLLoader.load(getClass().getResource("sceneRecognition.fxml"));
        Scene scene = new Scene(root);
        scene.getStylesheets().add(this.getClass().getResource("styles.css").toExternalForm());
        stage.setTitle("Digit Recognizer Based on Neural Network and Genetic Algorithm");
        stage.setScene(scene);
        stage.show();
    }

    public static void main(String[] args) {
        launch(args);
    }

}