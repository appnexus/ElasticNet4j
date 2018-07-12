package com.appnexus.opt.ml;

import java.io.Serializable;
import java.util.Arrays;
import java.util.List;

/**
 * This object represents the trained LR model for a specific set of training parameters. It also holds a list of {@link LRIterationMetadata} representing the intermediate results of each iteration during the training process
 */
public class LRResult implements Serializable {
    private static final long serialVersionUID = 4817212848479794019L;

    private double alpha;
    private double lambda;
    private double[] betasWithBeta0;
    private int iteration;
    private double maxAbsDifferencePct;
    private double trainingEntropy;
    private long trainingTimeMillis;

    // MetaData
    private List<LRIterationMetadata> metadataList;

    public double getAlpha() {
        return alpha;
    }

    public void setAlpha(double alpha) {
        this.alpha = alpha;
    }

    public double getLambda() {
        return lambda;
    }

    public void setLambda(double lambda) {
        this.lambda = lambda;
    }

    public double[] getBetasWithBeta0() {
        return betasWithBeta0;
    }

    public void setBetasWithBeta0(double[] betasWithBeta0) {
        this.betasWithBeta0 = betasWithBeta0;
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

    public List<LRIterationMetadata> getMetaDataList() {
        return metadataList;
    }

    public void setMetaDataList(List<LRIterationMetadata> metadataList) {
        this.metadataList = metadataList;
    }

    public long getTrainingTimeMillis() {
        return trainingTimeMillis;
    }

    public void setTrainingTimeMillis(long trainingTimeMillis) {
        this.trainingTimeMillis = trainingTimeMillis;
    }

    public double getTrainingEntropy() {
        return this.trainingEntropy;
    }

    public void setTrainingEntropy(double trainingEntropy) {
        this.trainingEntropy = trainingEntropy;
    }

    public static long getSerialversionuid() {
        return serialVersionUID;
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append("LRResult [alpha=");
        builder.append(this.alpha);
        builder.append(", lambda=");
        builder.append(this.lambda);
        builder.append(", betasWithBeta0=");
        builder.append(Arrays.toString(this.betasWithBeta0));
        builder.append(", iteration=");
        builder.append(this.iteration);
        builder.append(", maxAbsDifferencePct=");
        builder.append(this.maxAbsDifferencePct);
        builder.append(", trainingEntropy=");
        builder.append(this.trainingEntropy);
        builder.append(", trainingTimeMillis=");
        builder.append(this.trainingTimeMillis);
        builder.append(", metadataList=");
        builder.append(this.metadataList);
        builder.append("]");
        return builder.toString();
    }
}
