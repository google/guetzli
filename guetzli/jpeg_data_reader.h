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

// Functions for reading a jpeg byte stream into a JPEGData object.

#ifndef GUETZLI_JPEG_DATA_READER_H_
#define GUETZLI_JPEG_DATA_READER_H_

#include <stddef.h>
#include <stdint.h>

#include <string>

#include "guetzli/jpeg_data.h"

namespace guetzli {

enum JpegReadMode {
  JPEG_READ_HEADER,   // only basic headers
  JPEG_READ_TABLES,   // headers and tables (quant, Huffman, ...)
  JPEG_READ_ALL,      // everything
};

// Parses the jpeg stream contained in data[*pos ... len) and fills in *jpg with
// the parsed information.
// If mode is JPEG_READ_HEADER, it fills in only the image dimensions in *jpg.
// Returns false if the data is not valid jpeg, or if it contains an unsupported
// jpeg feature.
bool ReadJpeg(const uint8_t* data, const size_t len, JpegReadMode mode,
              JPEGData* jpg);
// string variant
bool ReadJpeg(const std::string& data, JpegReadMode mode,
              JPEGData* jpg);

}  // namespace guetzli

#endif  // GUETZLI_JPEG_DATA_READER_H_
