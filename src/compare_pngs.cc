#include <cmath>
#include <cstdio>
#include <vector>
#include "butteraugli.h"

extern "C" {
#include "png.h"
}

bool ReadPNG(FILE* f, std::vector<std::vector<double> >* rgb, int* xsize_out,
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
  rgb->resize(3);

  switch (components) {
    case 1: {
      // Indexcolor or gray-scale.
      png_bytep trans = 0;
      int num_trans = 0;
      png_color_16* trans_color = 0;
      png_get_tRNS(png_ptr, info_ptr, &trans, &num_trans, &trans_color);

      png_colorp palette;
      int num_palette;
      png_color_8_struct palette_with_alpha[256];
      if (png_get_PLTE(png_ptr, info_ptr, &palette, &num_palette)) {
        // We have a PLTE tag, the image is palettized.
        for (int i = 0; i < num_palette; ++i) {
          palette_with_alpha[i].red = palette[i].red;
          palette_with_alpha[i].green = palette[i].green;
          palette_with_alpha[i].blue = palette[i].blue;
          palette_with_alpha[i].alpha = 255;
        }
      } else {
        // We do not have a PLTE tag, the image is grayscale.
        for (int i = 0; i < 256; ++i) {
          palette_with_alpha[i].red = i;
          palette_with_alpha[i].green = i;
          palette_with_alpha[i].blue = i;
          palette_with_alpha[i].alpha = 255;
        }
      }
      for (int i = 0; i < num_trans; ++i) {
        palette_with_alpha[trans[i]].alpha = 0;
      }
      for (int y = 0; y < ysize; ++y) {
        for (int x = 0; x < xsize; ++x) {
          const int i = row_pointers[y][x];
          (*rgb)[0].push_back(palette_with_alpha[i].red);
          (*rgb)[1].push_back(palette_with_alpha[i].green);
          (*rgb)[2].push_back(palette_with_alpha[i].blue);
          // ALPHA
        }
      }
      break;
    }
    case 2:
      // Grayscale with alpha.
      for (int y = 0; y < ysize; ++y) {
        for (int x = 0; x < xsize; ++x) {
          (*rgb)[0].push_back(row_pointers[y][2 * x]);
          (*rgb)[1].push_back(row_pointers[y][2 * x]);
          (*rgb)[2].push_back(row_pointers[y][2 * x]);
          // ALPHA
        }
      }
      break;
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
          // ALPHA
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

void ApplyGamma(std::vector<std::vector<double> >* rgb, double gamma) {
  for (int c = 0; c < 3; c++) {
    for (size_t i = 0; i < (*rgb)[c].size(); i++) {
      (*rgb)[c][i] = 255.0 * pow((*rgb)[c][i] / 255.0, gamma);
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
  std::vector<std::vector<double> > rgb1, rgb2;
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
  // TODO: Figure out if it is a good idea to fetch the kGamma from the
  // image instead of having it hardcoded here.
  const double kGamma = 2.2;
  if (kGamma < 1.0) {
    fprintf(stderr,
            "Gamma is usually around 2.2, probably the gamma value "
            "should be inverted for reasonable butteraugli results");
    return 1;
  }
  ApplyGamma(&rgb1, kGamma);
  ApplyGamma(&rgb2, kGamma);
  std::vector<double> diffmap;
  double diffvalue;
  if (!butteraugli::ButteraugliInterface(xsize1, ysize1, rgb1, rgb2, diffmap,
                                         diffvalue)) {
    fprintf(stderr, "Butteraugli comparison failed\n");
    return 1;
  }
  printf("%lf\n", diffvalue);
  return 0;
}
