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

#ifndef GUETZLI_DEBUG_PRINT_H_
#define GUETZLI_DEBUG_PRINT_H_

#include "guetzli/stats.h"

namespace guetzli {

void PrintDebug(ProcessStats* stats, std::string s);

}  // namespace guetzli

#define GUETZLI_LOG(stats, ...)                                    \
  do {                                                             \
    char debug_string[1024];                                       \
    snprintf(debug_string, sizeof(debug_string), __VA_ARGS__);     \
    debug_string[sizeof(debug_string) - 1] = '\0';                 \
    ::guetzli::PrintDebug(                      \
         stats, std::string(debug_string));        \
  } while (0)
#define GUETZLI_LOG_QUANT(stats, q)                    \
  for (int y = 0; y < 8; ++y) {                        \
    for (int c = 0; c < 3; ++c) {                      \
      for (int x = 0; x < 8; ++x)                      \
        GUETZLI_LOG(stats, " %2d", (q)[c][8 * y + x]); \
      GUETZLI_LOG(stats, "   ");                       \
    }                                                  \
    GUETZLI_LOG(stats, "\n");                          \
  }

#endif  // GUETZLI_DEBUG_PRINT_H_
