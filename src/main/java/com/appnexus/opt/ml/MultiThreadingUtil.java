package com.appnexus.opt.ml;


import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.*;

class DatasetRange {
    private int startIdx;
    private int endIdx;
    private Object[] dataset;

    /**
     * class specifying subset of dataset with start and end indices
     * @param startIdx start index
     * @param endIdx end index
     * @param dataset entire dataset to operate on
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
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || this.getClass() != o.getClass()) return false;
        DatasetRange that = (DatasetRange) o;
        return this.startIdx == that.startIdx && this.endIdx == that.endIdx && Arrays
            .equals(this.dataset, that.dataset);
    }

    @Override
    public int hashCode() {
        int result = Objects.hash(this.startIdx, this.endIdx);
        result = 31 * result + Arrays.hashCode(this.dataset);
        return result;
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


public class MultiThreadingUtil {

    private static final int MT_EXEC_POOL_TIMEOUT_MS = 800;

    /**
     * wait for all threads pertaining to the completion service to complete
     *
     * @param completionService completion service
     * @param numThreads        number of threads
     */
    static void waitForThreadCompletion(CompletionService<Boolean> completionService, int numThreads) {
        int received = 0;
        while (received < numThreads) {
            try {
                completionService.take().get(); // blocks if none available
                received++;
            } catch (InterruptedException | ExecutionException e) {
                e.printStackTrace();
            }
        }
    }

    /**
     * @param dataset    dataset
     * @param numThreads number of threads
     * @return list of dataset ranges indicating indices by which to partition data
     */
    static List<DatasetRange> splitDatasetIntoRanges(Object[] dataset, int numThreads) {
        List<DatasetRange> datasetRanges = new LinkedList<>();
        int lengthOfDatasetRange = dataset.length / numThreads;
        int numDatasetRange = 0;
        while (numDatasetRange < numThreads) {
            int startIdx = numDatasetRange * lengthOfDatasetRange;
            int endIdx =
                numDatasetRange != numThreads - 1 ? (numDatasetRange + 1) * lengthOfDatasetRange : dataset.length;
            datasetRanges.add(new DatasetRange(startIdx, endIdx, dataset));
            numDatasetRange++;
        }
        return datasetRanges;
    }

    /**
     * @param completionService completion service
     * @param task              task
     */
    static void submitTask(CompletionService<Boolean> completionService, Callable<Boolean> task) {
        completionService.submit(task);
    }

    /**
     * @param execPool executor pool
     */
    public static void closeExecutorPool(ExecutorService execPool) {
        execPool.shutdown();
        try {
            if (!execPool.awaitTermination(MT_EXEC_POOL_TIMEOUT_MS, TimeUnit.MILLISECONDS)) {
                execPool.shutdownNow();
            }
        } catch (InterruptedException e) {
            execPool.shutdownNow();
        }
    }
}

