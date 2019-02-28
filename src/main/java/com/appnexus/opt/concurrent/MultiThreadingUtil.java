package com.appnexus.opt.concurrent;

import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.*;

public class MultiThreadingUtil {

    private static final int MT_EXEC_POOL_TIMEOUT_MS = 800;

    /**
     * wait for all threads pertaining to the completion service to complete
     *
     * @param completionService completion service
     * @param numThreads        number of threads
     */
    public static void waitForThreadCompletion(CompletionService<Boolean> completionService, int numThreads) {
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
    public static List<DatasetRange> splitDatasetIntoRanges(List<?> dataset, int numThreads) {
        List<DatasetRange> datasetRanges = new LinkedList<>();
        int lengthOfDatasetRange = dataset.size() / numThreads;
        int numDatasetRange = 0;
        while (numDatasetRange < numThreads) {
            int startIdx = numDatasetRange * lengthOfDatasetRange;
            int endIdx =
                numDatasetRange != numThreads - 1 ? (numDatasetRange + 1) * lengthOfDatasetRange : dataset.size();
            datasetRanges.add(new DatasetRange<>(startIdx, endIdx, dataset));
            numDatasetRange++;
        }
        return datasetRanges;
    }

    /**
     * @param completionService completion service
     * @param task              task
     */
    public static void submitTask(CompletionService<Boolean> completionService, Callable<Boolean> task) {
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

