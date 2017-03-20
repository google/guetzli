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
#include "png.h"
#include "guetzli/jpeg_data.h"
#include "guetzli/jpeg_data_reader.h"
#include "guetzli/processor.h"
#include "guetzli/quality.h"
#include "guetzli/stats.h"

namespace {

// An upper estimate of memory usage of Guetzli. The bound is
// max(kLowerMemusaeMB * 1<<20, pixel_count * kBytesPerPixel)
constexpr int kBytesPerPixel = 300;
constexpr int kLowestMemusageMB = 100; // in MB

constexpr int kDefaultMemlimitMB = 6000; // in MB

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
  fprintf(stderr, "Unhandled exception. Most likely insufficient memory available.\n"
          "Make sure that there is 300MB/MPix of memory available.\n");
  exit(1);
}

void Usage() {
  fprintf(stderr,
      "Guetzli JPEG compressor. Usage: \n"
      "guetzli [flags] input_filename output_filename\n"
      "\n"
      "Flags:\n"
      "  --verbose    - Print a verbose trace of all attempts to standard output.\n"
      "  --quality Q  - Visual quality to aim for, expressed as a JPEG quality value.\n"
      "  --memlimit M - Memory limit in MB. Guetzli will fail if unable to stay under\n"
      "                 the limit. Default is %d MB\n"
      "  --nomemlimit - Do not limit memory usage.\n", kDefaultMemlimitMB);
  exit(1);
}

}  // namespace

int main(int argc, char** argv) {
  std::set_terminate(TerminateHandler);

  int verbose = 0;
  int quality = 95;
  int memlimit_mb = kDefaultMemlimitMB;

  int opt_idx = 1;
  for(;opt_idx < argc;opt_idx++) {
    if (argv[opt_idx][0] != '-')
      break;
    if (!strcmp(argv[opt_idx], "--verbose")) {
      verbose = 1;
    } else if (!strcmp(argv[opt_idx], "--quality")) {
      opt_idx++;
      quality = atoi(argv[opt_idx]);
    } else if (!strcmp(argv[opt_idx], "--memlimit")) {
      opt_idx++;
      memlimit_mb = atoi(argv[opt_idx]);
    } else if (!strcmp(argv[opt_idx], "--nomemlimit")) {
      memlimit_mb = -1;
    } else {
      fprintf(stderr, "Unknown commandline flag: %s\n", argv[opt_idx]);
      Usage();
    }
  }

  if (argc - opt_idx != 2) {
    Usage();
  }

  FILE* fin = fopen(argv[opt_idx], "rb");
  if (!fin) {
    fprintf(stderr, "Can't open input file\n");
    return 1;
  }

  std::string in_data = ReadFileOrDie(fin);
  std::string out_data;

  guetzli::Params params;
  params.butteraugli_target =
      guetzli::ButteraugliScoreForQuality(quality);

  guetzli::ProcessStats stats;

  if (verbose) {
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
    int pixels = xsize * ysize;
    if (memlimit_mb != -1
        && (pixels * kBytesPerPixel / (1 << 20) > memlimit_mb
            || memlimit_mb < kLowestMemusageMB)) {
      fprintf(stderr, "Memory limit would be exceeded. Failing.\n");
      return 1;
    }
    if (!guetzli::Process(params, &stats, rgb, xsize, ysize, &out_data)) {
      fprintf(stderr, "Guetzli processing failed\n");
      return 1;
    }
  } else {
    guetzli::JPEGData jpg_header;
    if (!guetzli::ReadJpeg(in_data, guetzli::JPEG_READ_HEADER, &jpg_header)) {
      fprintf(stderr, "Error reading JPG data from input file\n");
      return 1;
    }
    int pixels = jpg_header.width * jpg_header.height;
    if (memlimit_mb != -1
        && (pixels * kBytesPerPixel / (1 << 20) > memlimit_mb
            || memlimit_mb < kLowestMemusageMB)) {
      fprintf(stderr, "Memory limit would be exceeded. Failing.\n");
      return 1;
    }
    if (!guetzli::Process(params, &stats, in_data, &out_data)) {
      fprintf(stderr, "Guetzli processing failed\n");
      return 1;
    }
  }

  fclose(fin);

  FILE* fout = fopen(argv[opt_idx + 1], "wb");
  if (!fout) {
    fprintf(stderr, "Can't open output file for writing\n");
    return 1;
  }

  WriteFileOrDie(fout, out_data);
  return 0;
}
