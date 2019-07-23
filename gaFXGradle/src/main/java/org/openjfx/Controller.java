package org.openjfx;

import org.openjfx.gaJava.MultiLayerNeuralNetworks.MultiLayerPerceptrons;
import javafx.application.Platform;
import javafx.fxml.FXML;
import javafx.scene.SnapshotParameters;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.Button;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Label;

import org.openjfx.gaJava.GeneticAlgorithms.GANeuralNetwork;
import javafx.scene.image.*;
import javafx.scene.image.Image;
import javafx.scene.input.MouseEvent;
import javafx.scene.layout.Pane;
import javafx.scene.paint.Color;
import javafx.stage.FileChooser;
import org.openjfx.gaJava.mnist.ImageManipulation;
import org.openjfx.gaJava.mnist.MnistMatrix;
import static org.openjfx.gaJava.util.ArrayTypeConversion.*;

import java.io.*;
import java.util.Arrays;

public class Controller {

    @FXML
    private Label digitRecognizedLabel;

    @FXML
    private Button recognizeDigitMlp;

    @FXML
    private Button recognizeDigitGA;

    @FXML
    private Pane digitDrawPane;

    @FXML
    private Button clear;

    @FXML
    private ComboBox comboMenu;

    // created dynamically and added to the digitDrawPane
    public Canvas canvas;

    // created dynamically and added to the digitDrawPane
    ImageView imageView;

    public static MnistMatrix[] inputMatrix;

    public static GANeuralNetwork ga;
    public static  MultiLayerPerceptrons mlp;

    private static SnapshotParameters SP = new SnapshotParameters();
    private static WritableImage WI = null;

    public void initialize() throws Exception {
        String javaVersion = System.getProperty("java.version");
        String javafxVersion = System.getProperty("javafx.version");
        System.out.println("java version " + javafxVersion + ", javaFX version " + javafxVersion);
        String javaProperties = "java version " + javafxVersion + ", javaFX version " + javafxVersion;
        Main.applicationStage.setTitle(Main.applicationStage.getTitle() + ", " + javaProperties);
        canvas = new Canvas(digitDrawPane.getPrefWidth(), digitDrawPane.getPrefHeight());

        WI = new WritableImage((int) canvas.getWidth(),
                (int) canvas.getHeight());

         ga = ga.getGANetwork();
         mlp = ga.getMlpNetwork();

        this.initializeAction();
    }

    public void initializeAction() {
        recognizeDigitGA.setOnAction((event) -> {
            try {
                SP = new SnapshotParameters();
                WritableImage image = digitDrawPane.snapshot(SP, WI);
                image = (WritableImage) scale(image, 28, 28, false);
                int[][] pixels = new int[(int) image.getWidth()][(int) image.getHeight()];

                for (int i = 0; i < image.getWidth(); i++) {
                    for (int j = 0; j < image.getHeight(); j++) {
                        int rgb = image.getPixelReader().getArgb(j, i);
                        int red = (rgb >> 16) & 0xff;
                        int green = (rgb >> 8) & 0xff;
                        int blue = rgb & 0xff;
                        pixels[i][j] = (red + green + blue) / 3;
                    }
                }

                ImageManipulation.saveIntToImage(pixels, "imageTest/test" + ".png");
                double[] doublePixels = normalizeMaxMin(convertIntToDouble(pixels));
                Integer[] predicted = ga.predict(doublePixels);
                int predictedOutput = Arrays.asList(predicted).indexOf(1);
                digitRecognizedLabel.setText("Digit recognized as " + predictedOutput);
            } catch (
                    Exception ex) {
                ex.printStackTrace();
            }
        });

        recognizeDigitMlp.setOnAction((event) -> {
            try {
                SP = new SnapshotParameters();
                WritableImage image = digitDrawPane.snapshot(SP, WI);
                image = (WritableImage) scale(image, 28, 28, false);
                int[][] pixels = new int[(int) image.getWidth()][(int) image.getHeight()];

                for (int i = 0; i < image.getWidth(); i++) {
                    for (int j = 0; j < image.getHeight(); j++) {
                        int rgb = image.getPixelReader().getArgb(j, i);
                        int red = (rgb >> 16) & 0xff;
                        int green = (rgb >> 8) & 0xff;
                        int blue = rgb & 0xff;
                        pixels[i][j] = (red + green + blue) / 3;
                    }
                }

                ImageManipulation.saveIntToImage(pixels, "imageTest/test" + ".png");
                double[] doublePixels = normalizeMaxMin(convertIntToDouble(pixels));

                Integer[] predicted = mlp.predict(doublePixels);
                int predictedOutput = Arrays.asList(predicted).indexOf(1);

                digitRecognizedLabel.setText("Digit recognized as " + predictedOutput);

               // GANeuralNetwork.evaluateModel();

//                String workingDir = System.getProperty("user.dir");
//                File file = new File(workingDir + "/test.png");
//
//                RenderedImage renderedImage = SwingFXUtils.fromFXImage(image, null);
//
//                ImageIO.write(
//                        renderedImage,
//                        "png",
//                        file);
//                System.out.println("image saved to " + workingDir + "/test.png");
            } catch (
                    Exception ex) {
                ex.printStackTrace();
            }
        });

        digitDrawPane.getChildren().add(canvas);
        canvas.setStyle("background-color : #ffaadd;");
        canvas.toFront();

        canvas.addEventHandler(MouseEvent.MOUSE_DRAGGED,
                (MouseEvent e) -> {
                    int radius = 17;
                    GraphicsContext gc = canvas.getGraphicsContext2D();
                    gc.setFill(Color.WHITE);
                    gc.fillOval(e.getX() - radius / 2, e.getY() - radius / 2, radius, radius);
                    gc.setStroke(Color.WHITE);
                });

        clear.setOnAction((event) -> {
            canvas.getGraphicsContext2D().clearRect(0, 0, digitDrawPane.getPrefWidth(),
                    digitDrawPane.getPrefHeight());
            digitDrawPane.getChildren().removeAll(imageView);
            imageView = null;
            digitRecognizedLabel.setText("");
        });

        comboMenu.setOnAction(e -> {
            System.out.println("combo clicked ");
            if (((ComboBox) e.getTarget()).getValue().equals("upload image")) {
                try {
                    uploadImage();
                } catch (Exception ex) {
                    ex.printStackTrace();
                }
            } else if (((ComboBox) e.getTarget()).getValue().equals("close")) {
                Platform.exit();
                System.exit(0);
            }
        });
    }

    public void uploadImage() throws Exception {
        FileChooser fileChooser = new FileChooser();
        fileChooser.setTitle("Open Resource File");
        File image = fileChooser.showOpenDialog(Main.applicationStage);

        if (image.exists()) {
            loadImageToPane(image);
        }
    }

    public Image scale(Image source, int targetWidth, int targetHeight, boolean preserveRatio) {
        ImageView imageView = new ImageView(source);
        imageView.setPreserveRatio(preserveRatio);
        imageView.setFitWidth(targetWidth);
        imageView.setFitHeight(targetHeight);
        return imageView.snapshot(null, null);
    }

    public void loadImageToPane(File imageFile) throws Exception {
        canvas.getGraphicsContext2D().clearRect(0, 0, digitDrawPane.getPrefWidth(),
                digitDrawPane.getPrefHeight());
        digitDrawPane.getChildren().remove(imageView);
        imageView = null;
        digitRecognizedLabel.setText("");

        Image image = new Image(new FileInputStream(imageFile.getPath()));
        image = scale(image, (int) canvas.getWidth(), (int) canvas.getHeight(), false);
        imageView = new ImageView(image);

        digitDrawPane.getChildren().add(imageView);

        comboMenu.setValue("File");
    }
}
