package com.appnexus.opt.ml;

public class ModelErrorMetrics {
    private double scaleFactor;
    private double predRatio;
    private double bias;
    private double entropyNormalized;

    public ModelErrorMetrics(double scaleFactor, double predRatio, double bias, double entropyNormalized) {
        this.scaleFactor = scaleFactor;
        this.predRatio = predRatio;
        this.bias = bias;
        this.entropyNormalized = entropyNormalized;
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
}

