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

#include "guetzli/debug_print.h"

namespace guetzli {

void PrintDebug(ProcessStats* stats, std::string s) {
  if (stats->debug_output) {
    stats->debug_output->append(s);
  }
  if (stats->debug_output_file) {
    fprintf(stats->debug_output_file, "%s", s.c_str());
  }
}

}  // namespace guetzli
