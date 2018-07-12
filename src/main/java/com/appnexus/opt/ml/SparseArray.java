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

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class SparseArray implements Iterable<SparseArray.Entry>, Serializable {

    private static final long serialVersionUID = 1L;

    public static class Entry implements Serializable {
        private static final long serialVersionUID = 1L;

        public int i;
        public double x;

        public Entry(int i, double x) {
            this.i = i;
            this.x = x;
        }

        @Override
        public String toString() {
            final StringBuilder sb = new StringBuilder("Entry{");
            sb.append("i=").append(i);
            sb.append(", x=").append(x);
            sb.append('}');
            return sb.toString();
        }
    }

    private List<Entry> array;

    public SparseArray() {
        this(10);
    }

    private SparseArray(int initialCapacity) {
        array = new ArrayList<>(initialCapacity);
    }

    public int size() {
        return array.size();
    }

    public boolean isEmpty() {
        return array.isEmpty();
    }

    @Override
    public Iterator<Entry> iterator() {
        return array.iterator();
    }

    public double get(int i) {
        for (Entry e : array) {
            if (e.i == i) {
                return e.x;
            }
        }

        return 0.0;
    }

    /**
     * Sets or add an entry.
     * 
     * @param i the index of entry.
     * @param x the value of entry.
     * @return true if a new entry added, false if an existing entry updated.
     */
    public boolean set(int i, double x) {
        if (x == 0.0) {
            remove(i);
            return false;
        }

        Iterator<Entry> it = array.iterator();
        for (int k = 0; it.hasNext(); k++) {
            Entry e = it.next();
            if (e.i == i) {
                e.x = x;
                return false;
            } else if (e.i > i) {
                array.add(k, new Entry(i, x));
                return true;
            }
        }

        array.add(new Entry(i, x));
        return true;
    }

    /**
     * Append an entry to the array, optimizing for the case where the index is greater than all existing indices in the array.
     * 
     * @param i the index of entry.
     * @param x the value of entry.
     */
    public void append(int i, double x) {
        if (x != 0.0) {
            array.add(new Entry(i, x));
        }
    }

    /**
     * Removes an entry.
     * 
     * @param i the index of entry.
     */
    public void remove(int i) {
        Iterator<Entry> it = array.iterator();
        while (it.hasNext()) {
            Entry e = it.next();
            if (e.i == i) {
                it.remove();
                break;
            }
        }
    }

    @Override
    public String toString() {
        final StringBuilder sb = new StringBuilder("SparseArray{");
        sb.append("array=").append(array);
        sb.append('}');
        return sb.toString();
    }
}
