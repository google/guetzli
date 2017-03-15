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

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <string>
#include <string.h>
#include <unistd.h>
#include "gflags/gflags.h"
#include "png.h"
#include "guetzli/processor.h"
#include "guetzli/quality.h"
#include "guetzli/stats.h"

#ifndef GFLAGS_NAMESPACE
using namespace gflags;
#else
using namespace GFLAGS_NAMESPACE;
#endif


DEFINE_bool(verbose, false,
            "Print a verbose trace of all attempts to standard output.");
DEFINE_double(quality, 95,
              "Visual quality to aim for, expressed as a JPEG quality value.");

namespace {

inline uint8_t BlendOnBlack(const uint8_t val, const uint8_t alpha) {
  return (static_cast<int>(val) * static_cast<int>(alpha) + 128) / 255;
}

bool ReadPNG(FILE* f, int* xsize, int* ysize,
             std::vector<uint8_t>* rgb) {
  png_structp png_ptr =
      png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  if (!png_ptr) {
    return false;
  }

  png_infop info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr) {
    png_destroy_read_struct(&png_ptr, nullptr, nullptr);
    return false;
  }

  if (setjmp(png_jmpbuf(png_ptr)) != 0) {
    // Ok we are here because of the setjmp.
    png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
    return false;
  }

  rewind(f);
  png_init_io(png_ptr, f);

  // The png_transforms flags are as follows:
  // packing == convert 1,2,4 bit images,
  // strip == 16 -> 8 bits / channel,
  // shift == use sBIT dynamics, and
  // expand == palettes -> rgb, grayscale -> 8 bit images, tRNS -> alpha.
  const unsigned int png_transforms =
      PNG_TRANSFORM_PACKING | PNG_TRANSFORM_EXPAND | PNG_TRANSFORM_STRIP_16;

  png_read_png(png_ptr, info_ptr, png_transforms, nullptr);

  png_bytep* row_pointers = png_get_rows(png_ptr, info_ptr);

  *xsize = png_get_image_width(png_ptr, info_ptr);
  *ysize = png_get_image_height(png_ptr, info_ptr);
  rgb->resize(3 * (*xsize) * (*ysize));

  const int components = png_get_channels(png_ptr, info_ptr);
  switch (components) {
    case 1: {
      // GRAYSCALE
      for (int y = 0; y < *ysize; ++y) {
        const uint8_t* row_in = row_pointers[y];
        uint8_t* row_out = &(*rgb)[3 * y * (*xsize)];
        for (int x = 0; x < *xsize; ++x) {
          const uint8_t gray = row_in[x];
          row_out[3 * x + 0] = gray;
          row_out[3 * x + 1] = gray;
          row_out[3 * x + 2] = gray;
        }
      }
      break;
    }
    case 2: {
      // GRAYSCALE + ALPHA
      for (int y = 0; y < *ysize; ++y) {
        const uint8_t* row_in = row_pointers[y];
        uint8_t* row_out = &(*rgb)[3 * y * (*xsize)];
        for (int x = 0; x < *xsize; ++x) {
          const uint8_t gray = BlendOnBlack(row_in[2 * x], row_in[2 * x + 1]);
          row_out[3 * x + 0] = gray;
          row_out[3 * x + 1] = gray;
          row_out[3 * x + 2] = gray;
        }
      }
      break;
    }
    case 3: {
      // RGB
      for (int y = 0; y < *ysize; ++y) {
        const uint8_t* row_in = row_pointers[y];
        uint8_t* row_out = &(*rgb)[3 * y * (*xsize)];
        memcpy(row_out, row_in, 3 * (*xsize));
      }
      break;
    }
    case 4: {
      // RGBA
      for (int y = 0; y < *ysize; ++y) {
        const uint8_t* row_in = row_pointers[y];
        uint8_t* row_out = &(*rgb)[3 * y * (*xsize)];
        for (int x = 0; x < *xsize; ++x) {
          const uint8_t alpha = row_in[4 * x + 3];
          row_out[3 * x + 0] = BlendOnBlack(row_in[4 * x + 0], alpha);
          row_out[3 * x + 1] = BlendOnBlack(row_in[4 * x + 1], alpha);
          row_out[3 * x + 2] = BlendOnBlack(row_in[4 * x + 2], alpha);
        }
      }
      break;
    }
    default:
      png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
      return false;
  }
  png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
  return true;
}

std::string ReadFileOrDie(FILE* f) {
  if (fseek(f, 0, SEEK_END) != 0) {
    perror("fseek");
    exit(1);
  }
  off_t size = ftell(f);
  if (size < 0) {
    perror("ftell");
    exit(1);
  }
  if (fseek(f, 0, SEEK_SET) != 0) {
    perror("fseek");
    exit(1);
  }
  std::unique_ptr<char[]> buf(new char[size]);
  if (fread(buf.get(), 1, size, f) != (size_t)size) {
    perror("fread");
    exit(1);
  }
  std::string result(buf.get(), size);
  return result;
}

void WriteFileOrDie(FILE* f, const std::string& contents) {
  if (fwrite(contents.data(), 1, contents.size(), f) != contents.size()) {
    perror("fwrite");
    exit(1);
  }
  if (fclose(f) < 0) {
    perror("fclose");
    exit(1);
  }
}

void TerminateHandler() {
  fprintf(stderr, "Unhandled expection. Most likely insufficient memory available.\n"
          "Make sure that there is 300MB/MPix of memory available.\n");
  _exit(1);
}

}  // namespace

int main(int argc, char** argv) {
  std::set_terminate(TerminateHandler);
  SetUsageMessage(
      "Guetzli JPEG compressor. Usage: \n"
      "guetzli [flags] input_filename output_filename");
  ParseCommandLineFlags(&argc, &argv, true);

  if (argc != 3) {
    ShowUsageWithFlags(argv[0]);
    return 1;
  }

  FILE* fin = fopen(argv[1], "rb");
  if (!fin) {
    fprintf(stderr, "Can't open input file\n");
    return 1;
  }

  std::string in_data = ReadFileOrDie(fin);
  std::string out_data;

  guetzli::Params params;
  params.butteraugli_target =
      guetzli::ButteraugliScoreForQuality(FLAGS_quality);

  guetzli::ProcessStats stats;

  if (FLAGS_verbose) {
    stats.debug_output_file = stdout;
  }

  static const unsigned char kPNGMagicBytes[] = {
      0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n',
  };
  if (in_data.size() >= 8 &&
      memcmp(in_data.data(), kPNGMagicBytes, sizeof(kPNGMagicBytes)) == 0) {
    int xsize, ysize;
    std::vector<uint8_t> rgb;
    if (!ReadPNG(fin, &xsize, &ysize, &rgb)) {
      fprintf(stderr, "Error reading PNG data from input file\n");
      return 1;
    }
    if (!guetzli::Process(params, &stats, rgb, xsize, ysize, &out_data)) {
      fprintf(stderr, "Guetzli processing failed\n");
      return 1;
    }
  } else {
    if (!guetzli::Process(params, &stats, in_data, &out_data)) {
      fprintf(stderr, "Guetzli processing failed\n");
      return 1;
    }
  }

  fclose(fin);

  FILE* fout = fopen(argv[2], "wb");
  if (!fout) {
    fprintf(stderr, "Can't open output file for writing\n");
    return 1;
  }

  WriteFileOrDie(fout, out_data);
  return 0;
}
