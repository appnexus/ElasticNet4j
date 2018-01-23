package com.appnexus.opt.ml;

import smile.math.SparseArray;

public class SparseObservation {
    private final SparseArray x;
    private final double y;
    private final int weight;

    public SparseObservation(SparseArray x, double y) {
        this.x = x;
        this.y = y;
        this.weight = 1;
    }

    public SparseObservation(SparseArray x, double y, int weight) {
        this.x = x;
        this.y = y;
        this.weight = weight;
    }

    public SparseArray getX() {
        return this.x;
    }

    public int getWeight() {
        return this.weight;
    }

    public double getY() {
        return this.y;
    }
}
