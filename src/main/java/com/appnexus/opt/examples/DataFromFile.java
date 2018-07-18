package com.appnexus.opt.examples;

import com.appnexus.opt.ml.SparseObservation;

/*
 * This POJO contains all relevant data retrieved from a file that is
 * necessary for training and testing the logistic regression algorithm.
 */
public class DataFromFile {

    private SparseObservation[] sparseObservations;
    private int numOfFeatures;

    public DataFromFile(SparseObservation[] sparseObservations, int numOfFeatures) {
        this.sparseObservations = sparseObservations;
        this.numOfFeatures = numOfFeatures;
    }

    public SparseObservation[] getSparseObservations() {
        return sparseObservations;
    }

    public int getNumOfFeatures() {
        return numOfFeatures;
    }
}
