/*
 * Copyright 2016 Jiang Chen <criver@gmail.com>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "datastore_py.h"
#include "forest_py.h"
#include "gbdt_py_base.h"
#include "train_gbdt_py.h"

PYBIND11_PLUGIN(libgbdt) {
    py::module m("libgbdt", "GBDT python library");
    InitBinnedFloatColumnPy(m);
    InitRawFloatColumnPy(m);
    InitStringColumnPy(m);
    InitDataStorePy(m);
    InitForestPy(m);
    InitTrainGBDTPy(m);
    return m.ptr();
}
