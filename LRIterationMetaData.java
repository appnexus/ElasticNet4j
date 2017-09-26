package com.appnexus.opt.ml;

public class LRIterationMetaData {
    private double alpha;
    private double lambda;
    private int iteration;
    private double maxAbsDifferencePct;
    private double[] betas;
    private long trainingTimeMillis;

    public double getAlpha() {
        return alpha;
    }

    public void setAlpha(double alpha) {
        this.alpha = alpha;
    }

    public double getLamda() {
        return lambda;
    }

    public void setLamda(double lamda) {
        this.lambda = lamda;
    }

    public int getIteration() {
        return iteration;
    }

    public void setIteration(int iteration) {
        this.iteration = iteration;
    }

    public double getMaxAbsDifferencePct() {
        return maxAbsDifferencePct;
    }

    public void setMaxAbsDifferencePct(double maxAbsDifferencePct) {
        this.maxAbsDifferencePct = maxAbsDifferencePct;
    }

    public double[] getBetas() {
        return betas;
    }

    public void setBetas(double[] betas) {
        this.betas = betas;
    }

    public long getTrainingTimeMillis() {
        return trainingTimeMillis;
    }

    public void setTrainingTimeMillis(long trainingTimeMillis) {
        this.trainingTimeMillis = trainingTimeMillis;
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append("LRMetaData [lamda=");
        builder.append(lambda);
        builder.append(", alpha=");
        builder.append(alpha);
        builder.append(", iteration=");
        builder.append(iteration);
        builder.append(", maxAbsDifferencePct=");
        builder.append(maxAbsDifferencePct);
        // builder.append(", betas=");
        // builder.append(Arrays.toString(betas));
        builder.append(", training time (millis)=");
        builder.append(trainingTimeMillis);
        builder.append("]");
        return builder.toString();
    }
}
