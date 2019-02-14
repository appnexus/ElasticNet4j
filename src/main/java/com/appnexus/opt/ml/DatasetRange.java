package com.appnexus.opt.ml;

import java.util.Arrays;
import java.util.Objects;

public class DatasetRange {
    private int startIdx;
    private int endIdx;
    private Object[] dataset;

    /**
     * class specifying subset of dataset with start and end indices
     *
     * @param startIdx start index
     * @param endIdx   end index
     * @param dataset  entire dataset to operate on
     */
    DatasetRange(int startIdx, int endIdx, Object[] dataset) {
        this.startIdx = startIdx;
        this.endIdx = endIdx;
        this.dataset = dataset;
    }

    int getStartIdx() {
        return this.startIdx;
    }

    int getEndIdx() {
        return this.endIdx;
    }

    Object[] getDataset() {
        return this.dataset;
    }

    @Override
    public int hashCode() {
        int result = Objects.hash(this.startIdx, this.endIdx);
        result = 31 * result + Arrays.hashCode(this.dataset);
        return result;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || this.getClass() != o.getClass()) return false;
        DatasetRange that = (DatasetRange) o;
        return this.startIdx == that.startIdx && this.endIdx == that.endIdx && Arrays
            .equals(this.dataset, that.dataset);
    }

    @Override
    public String toString() {
        final StringBuilder sb = new StringBuilder("DatasetRange{");
        sb.append("startIdx=").append(startIdx);
        sb.append(", endIdx=").append(endIdx);
        sb.append(", dataset=").append(Arrays.toString(dataset));
        sb.append('}');
        return sb.toString();
    }
}
