package com.appnexus.opt.concurrent;

import java.util.Collections;
import java.util.List;
import java.util.Objects;

public class DatasetRange<T> {
    private int startIdx;
    private int endIdx;
    private List<T> dataset;

    /**
     * class specifying subset of dataset with start and end indices
     *
     * @param startIdx start index
     * @param endIdx   end index
     * @param dataset  entire dataset to operate on
     */
    public DatasetRange(int startIdx, int endIdx, List<T> dataset) {
        this.startIdx = startIdx;
        this.endIdx = endIdx;
        this.dataset = Collections.unmodifiableList(dataset);
    }

    public int getStartIdx() {
        return this.startIdx;
    }

    public int getEndIdx() {
        return this.endIdx;
    }

    public List<T> getDataset() {
        return this.dataset;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        DatasetRange<?> that = (DatasetRange<?>) o;
        return startIdx == that.startIdx && endIdx == that.endIdx && dataset.equals(that.dataset);
    }

    @Override
    public int hashCode() {
        return Objects.hash(startIdx, endIdx, dataset);
    }

    @Override
    public String toString() {
        final StringBuilder sb = new StringBuilder("DatasetRange{");
        sb.append("startIdx=").append(startIdx);
        sb.append(", endIdx=").append(endIdx);
        sb.append(", dataset=").append(dataset);
        sb.append('}');
        return sb.toString();
    }
}
