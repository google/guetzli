/*
 * Copyright 2016 Google Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef GUETZLI_PROCESSOR_H_
#define GUETZLI_PROCESSOR_H_

#include <string>
#include <vector>

#include "guetzli/comparator.h"
#include "guetzli/jpeg_data.h"
#include "guetzli/stats.h"

namespace guetzli {

struct Params {
  float butteraugli_target = 1.0;
  bool clear_metadata = false;
  bool try_420 = false;
  bool force_420 = false;
  bool use_silver_screen = false;
  int zeroing_greedy_lookahead = 3;
  bool new_zeroing_model = true;
};

bool Process(const Params& params, ProcessStats* stats,
             const std::string& in_data,
             std::string* out_data);

struct GuetzliOutput {
  std::string jpeg_data;
  std::vector<float> distmap;
  double distmap_aggregate;
  double score;
};

bool ProcessJpegData(const Params& params, const JPEGData& jpg_in,
                     Comparator* comparator, GuetzliOutput* out,
                     ProcessStats* stats);

// Sets *out to a jpeg encoded string that will decode to an image that is
// visually indistinguishable from the input rgb image.
bool Process(const Params& params, ProcessStats* stats,
             const std::vector<uint8_t>& rgb, int w, int h,
             std::string* out);

}  // namespace guetzli

#endif  // GUETZLI_PROCESSOR_H_
