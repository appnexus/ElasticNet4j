package com.appnexus.opt.ml;

import java.util.Arrays;

public class LRIterationMetadata {
    private double alpha;
    private double lambda;
    private int iteration;
    private double maxAbsDifferencePct;
    private double trainingEntropy;
    private double[] betas;
    private long trainingTimeMillis;

    public double getAlpha() {
        return this.alpha;
    }

    public void setAlpha(double alpha) {
        this.alpha = alpha;
    }

    public double getLambda() {
        return this.lambda;
    }

    public void setLambda(double lambda) {
        this.lambda = lambda;
    }

    public int getIteration() {
        return this.iteration;
    }

    public void setIteration(int iteration) {
        this.iteration = iteration;
    }

    public double getMaxAbsDifferencePct() {
        return this.maxAbsDifferencePct;
    }

    public void setMaxAbsDifferencePct(double maxAbsDifferencePct) {
        this.maxAbsDifferencePct = maxAbsDifferencePct;
    }

    public double getTrainingEntropy() {
        return this.trainingEntropy;
    }

    public void setTrainingEntropy(double trainingEntropy) {
        this.trainingEntropy = trainingEntropy;
    }

    public double[] getBetas() {
        return this.betas;
    }

    public void setBetas(double[] betas) {
        this.betas = betas;
    }

    public long getTrainingTimeMillis() {
        return this.trainingTimeMillis;
    }

    public void setTrainingTimeMillis(long trainingTimeMillis) {
        this.trainingTimeMillis = trainingTimeMillis;
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append("LRIterationMetadata [alpha=");
        builder.append(this.alpha);
        builder.append(", lambda=");
        builder.append(this.lambda);
        builder.append(", iteration=");
        builder.append(this.iteration);
        builder.append(", maxAbsDifferencePct=");
        builder.append(this.maxAbsDifferencePct);
        builder.append(", trainingEntropy=");
        builder.append(this.trainingEntropy);
        builder.append(", betas=");
        builder.append(Arrays.toString(this.betas));
        builder.append(", trainingTimeMillis=");
        builder.append(this.trainingTimeMillis);
        builder.append("]");
        return builder.toString();
    }
}
