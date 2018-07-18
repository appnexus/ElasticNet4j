package com.appnexus.opt.examples;

import com.appnexus.opt.ml.SparseObservation;

// TODO: documentation?
class DataFromFile {

    private SparseObservation[] sparseObservations;
    private int numOfFeatures;

    DataFromFile(SparseObservation[] sparseObservations, int numOfFeatures) {
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
