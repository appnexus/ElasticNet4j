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

/**
 * This class implements data structure representing a sparsely populated feature vector with its weight
 */
public class SparseObservation {
    private final SparseArray x;
    private final double y;
    private final int weight;

    public SparseObservation(SparseArray x, double y) {
        this(x, y, 1);
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

    @Override
    public String toString() {
        final StringBuilder sb = new StringBuilder("SparseObservation{");
        sb.append("x=").append(x);
        sb.append(", y=").append(y);
        sb.append(", weight=").append(weight);
        sb.append('}');
        return sb.toString();
    }
}
