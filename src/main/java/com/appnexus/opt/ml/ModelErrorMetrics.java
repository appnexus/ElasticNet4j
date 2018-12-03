package com.appnexus.opt.ml;

public class ModelErrorMetrics {
    private double scaleFactor;
    private double predRatio;
    private double bias;
    private double entropyNormalized;
    private boolean chosenModel;

    public ModelErrorMetrics(double scaleFactor, double predRatio, double bias, double entropyNormalized,
        boolean chosenModel) {
        this.scaleFactor = scaleFactor;
        this.predRatio = predRatio;
        this.bias = bias;
        this.entropyNormalized = entropyNormalized;
        this.chosenModel = chosenModel;
    }

    public double getScaleFactor() {
        return this.scaleFactor;
    }

    public double getPredRatio() {
        return this.predRatio;
    }

    public double getBias() {
        return this.bias;
    }

    public double getEntropyNormalized() {
        return this.entropyNormalized;
    }

    public boolean isChosenModel() {
        return this.chosenModel;
    }
}

