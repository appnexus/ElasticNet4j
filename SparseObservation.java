package com.appnexus.opt.ml;

import smile.math.SparseArray;

public class SparseObservation {
    private final SparseArray x;
    private final int weight;
    private final double y;

    public SparseObservation(SparseArray x, double y) {
        this.x = x;
        this.weight = 1;
        this.y = y;
    }

    public SparseObservation(SparseArray x, int weight, double y) {
        this.x = x;
        this.weight = weight;
        this.y = y;
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
