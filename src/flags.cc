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

#include <gflags/gflags.h>

DEFINE_string(tsvs, "", "The comma separated tsv files. The first tsv contains the header.");
DEFINE_string(flatfiles_dirs, "", "The flatfiles dir.");
DEFINE_string(training_weight_file, "", "The training weight file.");
DEFINE_string(output_dir, "", "The output dir.");
DEFINE_string(output_model_name, "forest", "The output model name.");
DEFINE_string(testing_model_file, "", "The testing model file.");
DEFINE_string(base_model_file, "", "The base model file.");
DEFINE_string(config_file, "", "The config file.");
DEFINE_int32(num_threads, 16, "The number of threads.");
DEFINE_string(mode, "train", "The running mode.");
DEFINE_int32(seed, 1234567, "The random seed.");
