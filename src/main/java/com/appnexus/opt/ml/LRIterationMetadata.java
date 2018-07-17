/*
 *    Copyright 2018 APPNEXUS INC
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package com.appnexus.opt.ml;

import java.util.Arrays;

/**
 * This object holds the intermediate results and metadata related to a specific iteration in the model training process
 */
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
