#include <stdint.h>
#include <cmath>
#include <cstdio>
#include <vector>
#include "butteraugli.h"

extern "C" {
#include "png.h"
}

bool ReadPNG(FILE* f, std::vector<std::vector<uint8_t> >* rgb, int* xsize_out,
             int* ysize_out) {
  png_structp png_ptr =
      png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  if (!png_ptr) {
    return false;
  }

  png_infop info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr) {
    png_destroy_read_struct(&png_ptr, NULL, NULL);
    return false;
  }

  if (setjmp(png_jmpbuf(png_ptr)) != 0) {
    // Ok we are here because of the setjmp.
    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
    return false;
  }

  png_init_io(png_ptr, f);

  // The png_transforms flags are as follows:
  // packing == convert 1,2,4 bit images,
  // strip == 16 -> 8 bits / channel,
  // shift == use sBIT dynamics, and
  // expand == palettes -> rgb, grayscale -> 8 bit images, tRNS -> alpha.
  const unsigned int png_transforms =
      PNG_TRANSFORM_PACKING | PNG_TRANSFORM_EXPAND | PNG_TRANSFORM_STRIP_16;

  png_read_png(png_ptr, info_ptr, png_transforms, NULL);

  png_bytep* row_pointers = png_get_rows(png_ptr, info_ptr);

  const int xsize = png_get_image_width(png_ptr, info_ptr);
  *xsize_out = xsize;
  const int ysize = png_get_image_height(png_ptr, info_ptr);
  *ysize_out = ysize;
  const int components = png_get_channels(png_ptr, info_ptr);

  rgb->clear();
  rgb->resize(components);

  switch (components) {
    case 3: {
      // RGB
      for (int y = 0; y < ysize; ++y) {
        for (int x = 0; x < xsize; ++x) {
          (*rgb)[0].push_back(row_pointers[y][3 * x + 0]);
          (*rgb)[1].push_back(row_pointers[y][3 * x + 1]);
          (*rgb)[2].push_back(row_pointers[y][3 * x + 2]);
        }
      }
      break;
    }
    case 4: {
      // RGBA
      for (int y = 0; y < ysize; ++y) {
        for (int x = 0; x < xsize; ++x) {
          (*rgb)[0].push_back(row_pointers[y][4 * x + 0]);
          (*rgb)[1].push_back(row_pointers[y][4 * x + 1]);
          (*rgb)[2].push_back(row_pointers[y][4 * x + 2]);
          (*rgb)[3].push_back(row_pointers[y][4 * x + 3]);
        }
      }
      break;
    }
    default:
      png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
      return false;
  }
  png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
  return true;
}

const double* NewSrgbToLinearTable() {
  double* table = new double[256];
  for (int i = 0; i < 256; ++i) {
    const double srgb = i / 255.0;
    table[i] =
        255.0 * (srgb <= 0.04045 ? srgb / 12.92
                                 : std::pow((srgb + 0.055) / 1.055, 2.4));
  }
  return table;
}

// Translate R, G, B channels from sRGB to linear space. If an alpha channel
// is present, overlay the image over a black or white background. Overlaying
// is done in the sRGB space; while technically incorrect, this is aligned with
// many other software (web browsers, WebP near lossless).
void FromSrgbToLinear(const std::vector<std::vector<uint8_t> >& rgb,
                      std::vector<std::vector<float> >& linear,
                      int background) {
  static const double* const kSrgbToLinearTable = NewSrgbToLinearTable();
  linear.resize(3);
  for (int c = 0; c < 3; c++) {
    linear[c].resize(rgb[c].size());
    for (size_t i = 0; i < rgb[c].size(); i++) {
      int value;
      if (rgb.size() == 3 || rgb[3][i] == 255) {
        value = rgb[c][i];
      } else if (rgb[3][i] == 0) {
        value = background;
      } else {
        const int fg_weight = rgb[3][i];
        const int bg_weight = 255 - fg_weight;
        value = (rgb[c][i] * fg_weight + background * bg_weight + 127) / 255;
      }
      linear[c][i] = kSrgbToLinearTable[value];
    }
  }
}

int main(int argc, char** argv) {
  if (argc != 3) {
    fprintf(stderr, "Usage: %s {image1.png} {image2.png}\n", argv[0]);
    return 1;
  }
  FILE* f1 = fopen(argv[1], "r");
  if (!f1) {
    fprintf(stderr, "Cannot open %s\n", argv[1]);
    return 1;
  }
  FILE* f2 = fopen(argv[2], "r");
  if (!f2) {
    fprintf(stderr, "Cannot open %s\n", argv[2]);
    return 1;
  }
  std::vector<std::vector<uint8_t> > rgb1, rgb2;
  int xsize1, ysize1, xsize2, ysize2;
  if (!ReadPNG(f1, &rgb1, &xsize1, &ysize1)) {
    fprintf(stderr, "Cannot parse PNG file %s\n", argv[1]);
    return 1;
  }
  if (!ReadPNG(f2, &rgb2, &xsize2, &ysize2)) {
    fprintf(stderr, "Cannot parse PNG file %s\n", argv[2]);
    return 1;
  }
  if (xsize1 != xsize2 || ysize1 != ysize2) {
    fprintf(stderr, "The images are not equal in size: (%d,%d) vs (%d,%d)\n",
            xsize1, ysize1, xsize2, ysize2);
    return 1;
  }
  // TODO: Figure out if it is a good idea to fetch the gamma from the image
  // instead of applying sRGB conversion.
  std::vector<std::vector<float> > linear1, linear2;
  // Overlay the image over a black background.
  FromSrgbToLinear(rgb1, linear1, 0);
  FromSrgbToLinear(rgb2, linear2, 0);
  std::vector<float> diff_map;
  double diff_value;
  if (!butteraugli::ButteraugliInterface(xsize1, ysize1, linear1, linear2,
                                         diff_map, diff_value)) {
    fprintf(stderr, "Butteraugli comparison failed\n");
    return 1;
  }
  if (rgb1.size() == 4 || rgb2.size() == 4) {
    // If the alpha channel is present, overlay the image over a white
    // background as well.
    FromSrgbToLinear(rgb1, linear1, 255);
    FromSrgbToLinear(rgb2, linear2, 255);
    double diff_value_on_white;
    if (!butteraugli::ButteraugliInterface(xsize1, ysize1, linear1, linear2,
                                           diff_map, diff_value_on_white)) {
      fprintf(stderr, "Butteraugli comparison failed\n");
      return 1;
    }
    if (diff_value_on_white > diff_value) diff_value = diff_value_on_white;
  }
  printf("%lf\n", diff_value);
  return 0;
}
