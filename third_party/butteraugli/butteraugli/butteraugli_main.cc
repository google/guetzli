#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>
#include "butteraugli/butteraugli.h"

extern "C" {
#include "png.h"
#include "jpeglib.h"
}

namespace butteraugli {
namespace {

// "rgb": cleared and filled with same-sized image planes (one per channel);
// either RGB, or RGBA if the PNG contains an alpha channel.
bool ReadPNG(FILE* f, std::vector<Image8>* rgb) {
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

  rewind(f);
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
  const int ysize = png_get_image_height(png_ptr, info_ptr);
  const int components = png_get_channels(png_ptr, info_ptr);

  *rgb = CreatePlanes<uint8_t>(xsize, ysize, 3);

  switch (components) {
    case 1: {
      // GRAYSCALE
      for (int y = 0; y < ysize; ++y) {
        ConstRestrict<const uint8_t*> row = row_pointers[y];
        ConstRestrict<uint8_t*> row0 = (*rgb)[0].Row(y);
        ConstRestrict<uint8_t*> row1 = (*rgb)[1].Row(y);
        ConstRestrict<uint8_t*> row2 = (*rgb)[2].Row(y);

        for (int x = 0; x < xsize; ++x) {
          const uint8_t gray = row[x];
          row0[x] = row1[x] = row2[x] = gray;
        }
      }
      break;
    }
    case 2: {
      // GRAYSCALE_ALPHA
      rgb->push_back(Image8(xsize, ysize));
      for (int y = 0; y < ysize; ++y) {
        ConstRestrict<const uint8_t*> row = row_pointers[y];
        ConstRestrict<uint8_t*> row0 = (*rgb)[0].Row(y);
        ConstRestrict<uint8_t*> row1 = (*rgb)[1].Row(y);
        ConstRestrict<uint8_t*> row2 = (*rgb)[2].Row(y);
        ConstRestrict<uint8_t*> row3 = (*rgb)[3].Row(y);

        for (int x = 0; x < xsize; ++x) {
          const uint8_t gray = row[2 * x + 0];
          const uint8_t alpha = row[2 * x + 1];
          row0[x] = gray;
          row1[x] = gray;
          row2[x] = gray;
          row3[x] = alpha;
        }
      }
      break;
    }
    case 3: {
      // RGB
      for (int y = 0; y < ysize; ++y) {
        ConstRestrict<const uint8_t*> row = row_pointers[y];
        ConstRestrict<uint8_t*> row0 = (*rgb)[0].Row(y);
        ConstRestrict<uint8_t*> row1 = (*rgb)[1].Row(y);
        ConstRestrict<uint8_t*> row2 = (*rgb)[2].Row(y);

        for (int x = 0; x < xsize; ++x) {
          row0[x] = row[3 * x + 0];
          row1[x] = row[3 * x + 1];
          row2[x] = row[3 * x + 2];
        }
      }
      break;
    }
    case 4: {
      // RGBA
      rgb->push_back(Image8(xsize, ysize));
      for (int y = 0; y < ysize; ++y) {
        ConstRestrict<const uint8_t*> row = row_pointers[y];
        ConstRestrict<uint8_t*> row0 = (*rgb)[0].Row(y);
        ConstRestrict<uint8_t*> row1 = (*rgb)[1].Row(y);
        ConstRestrict<uint8_t*> row2 = (*rgb)[2].Row(y);
        ConstRestrict<uint8_t*> row3 = (*rgb)[3].Row(y);

        for (int x = 0; x < xsize; ++x) {
          row0[x] = row[4 * x + 0];
          row1[x] = row[4 * x + 1];
          row2[x] = row[4 * x + 2];
          row3[x] = row[4 * x + 3];
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

void jpeg_catch_error(j_common_ptr cinfo) {
  (*cinfo->err->output_message) (cinfo);
  jmp_buf* jpeg_jmpbuf = (jmp_buf*) cinfo->client_data;
  jpeg_destroy(cinfo);
  longjmp(*jpeg_jmpbuf, 1);
}

// "rgb": cleared and filled with same-sized image planes (one per channel);
// either RGB, or RGBA if the PNG contains an alpha channel.
bool ReadJPEG(FILE* f, std::vector<Image8>* rgb) {
  rewind(f);

  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;
  cinfo.err = jpeg_std_error(&jerr);
  jmp_buf jpeg_jmpbuf;
  cinfo.client_data = &jpeg_jmpbuf;
  jerr.error_exit = jpeg_catch_error;
  if (setjmp(jpeg_jmpbuf)) {
    return false;
  }

  jpeg_create_decompress(&cinfo);

  jpeg_stdio_src(&cinfo, f);
  jpeg_read_header(&cinfo, TRUE);
  jpeg_start_decompress(&cinfo);

  int row_stride = cinfo.output_width * cinfo.output_components;
  JSAMPARRAY buffer = (*cinfo.mem->alloc_sarray)
    ((j_common_ptr) &cinfo, JPOOL_IMAGE, row_stride, 1);

  const size_t xsize = cinfo.output_width;
  const size_t ysize = cinfo.output_height;

  *rgb = CreatePlanes<uint8_t>(xsize, ysize, 3);

  switch (cinfo.out_color_space) {
    case JCS_GRAYSCALE:
      while (cinfo.output_scanline < cinfo.output_height) {
        jpeg_read_scanlines(&cinfo, buffer, 1);

        ConstRestrict<const uint8_t*> row = buffer[0];
        ConstRestrict<uint8_t*> row0 = (*rgb)[0].Row(cinfo.output_scanline - 1);
        ConstRestrict<uint8_t*> row1 = (*rgb)[1].Row(cinfo.output_scanline - 1);
        ConstRestrict<uint8_t*> row2 = (*rgb)[2].Row(cinfo.output_scanline - 1);

        for (int x = 0; x < xsize; x++) {
          const uint8_t gray = row[x];
          row0[x] = row1[x] = row2[x] = gray;
        }
      }
      break;

    case JCS_RGB:
      while (cinfo.output_scanline < cinfo.output_height) {
        jpeg_read_scanlines(&cinfo, buffer, 1);

        ConstRestrict<const uint8_t*> row = buffer[0];
        ConstRestrict<uint8_t*> row0 = (*rgb)[0].Row(cinfo.output_scanline - 1);
        ConstRestrict<uint8_t*> row1 = (*rgb)[1].Row(cinfo.output_scanline - 1);
        ConstRestrict<uint8_t*> row2 = (*rgb)[2].Row(cinfo.output_scanline - 1);

        for (int x = 0; x < xsize; x++) {
          row0[x] = row[3 * x + 0];
          row1[x] = row[3 * x + 1];
          row2[x] = row[3 * x + 2];
        }
      }
      break;

    default:
      jpeg_destroy_decompress(&cinfo);
      return false;
  }

  jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);
  return true;
}

// Translate R, G, B channels from sRGB to linear space. If an alpha channel
// is present, overlay the image over a black or white background. Overlaying
// is done in the sRGB space; while technically incorrect, this is aligned with
// many other software (web browsers, WebP near lossless).
void FromSrgbToLinear(const std::vector<Image8>& rgb,
                      std::vector<ImageF>& linear, int background) {
  const size_t xsize = rgb[0].xsize();
  const size_t ysize = rgb[0].ysize();
  static const double* const kSrgbToLinearTable = NewSrgbToLinearTable();

  if (rgb.size() == 3) {  // RGB
    for (int c = 0; c < 3; c++) {
      linear.push_back(ImageF(xsize, ysize));
      for (int y = 0; y < ysize; ++y) {
        ConstRestrict<const uint8_t*> row_rgb = rgb[c].Row(y);
        ConstRestrict<float*> row_linear = linear[c].Row(y);
        for (size_t x = 0; x < xsize; x++) {
          const int value = row_rgb[x];
          row_linear[x] = kSrgbToLinearTable[value];
        }
      }
    }
  } else {  // RGBA
    for (int c = 0; c < 3; c++) {
      linear.push_back(ImageF(xsize, ysize));
      for (int y = 0; y < ysize; ++y) {
        ConstRestrict<const uint8_t*> row_rgb = rgb[c].Row(y);
        ConstRestrict<float*> row_linear = linear[c].Row(y);
        ConstRestrict<const uint8_t*> row_alpha = rgb[3].Row(y);
        for (size_t x = 0; x < xsize; x++) {
          int value;
          if (row_alpha[x] == 255) {
            value = row_rgb[x];
          } else if (row_alpha[x] == 0) {
            value = background;
          } else {
            const int fg_weight = row_alpha[x];
            const int bg_weight = 255 - fg_weight;
            value =
                (row_rgb[x] * fg_weight + background * bg_weight + 127) / 255;
          }
          row_linear[x] = kSrgbToLinearTable[value];
        }
      }
    }
  }
}

std::vector<Image8> ReadImageOrDie(const char* filename) {
  std::vector<Image8> rgb;
  FILE* f = fopen(filename, "r");
  if (!f) {
    fprintf(stderr, "Cannot open %s\n", filename);
    exit(1);
  }
  unsigned char magic[2];
  if (fread(magic, 1, 2, f) != 2) {
    fprintf(stderr, "Cannot read from %s\n", filename);
    exit(1);
  }
  if (magic[0] == 0xFF && magic[1] == 0xD8) {
    if (!ReadJPEG(f, &rgb)) {
      fprintf(stderr, "File %s is a malformed JPEG.\n", filename);
      exit(1);
    }
  } else {
    if (!ReadPNG(f, &rgb)) {
      fprintf(stderr, "File %s is neither a valid JPEG nor a valid PNG.\n",
              filename);
      exit(1);
    }
  }
  fclose(f);
  return rgb;
}

void CreateHeatMapImage(const ImageF& distmap, double good_threshold,
                        double bad_threshold, size_t xsize, size_t ysize,
                        std::vector<uint8_t>* heatmap) {
  heatmap->resize(3 * xsize * ysize);
  for (size_t y = 0; y < ysize; ++y) {
    for (size_t x = 0; x < xsize; ++x) {
      int px = xsize * y + x;
      double d = distmap.Row(y)[x];
      uint8_t* rgb = &(*heatmap)[3 * px];
      if (d < 0.5 * good_threshold) {
        rgb[0] = 0;
        rgb[1] = d * (255 - 153) / (0.5 * good_threshold) + 153;
        rgb[2] = 0;
      } else if (d < good_threshold) {
        d -= 0.5 * good_threshold;
        rgb[0] = d * 255 / (0.5 * good_threshold);
        rgb[1] = 255;
        rgb[2] = 0;
      } else if (d < bad_threshold) {
        d -= good_threshold;
        rgb[0] = 255;
        rgb[1] = 255 - d * 255 / (bad_threshold - good_threshold);
        rgb[2] = 0;
      } else if (d < 5 * bad_threshold) {
        rgb[0] = 255;
        rgb[1] = 0;
        rgb[2] = 255 * (d - bad_threshold) / (4 * bad_threshold);
      } else if (d < 10 * bad_threshold) {
        rgb[0] = 255;
        rgb[1] = 255 * (d - 5 * bad_threshold) / (5 * bad_threshold);
        rgb[2] = 255;
      } else {
        rgb[0] = 255;
        rgb[1] = 255;
        rgb[2] = 255;
      }
    }
  }
}

// main() function, within butteraugli namespace for convenience.
int Run(int argc, char* argv[]) {
  if (argc != 3 && argc != 4) {
    fprintf(stderr,
            "Usage: %s {image1.(png|jpg|jpeg)} {image2.(png|jpg|jpeg)} "
            "[heatmap.ppm]\n",
            argv[0]);
    return 1;
  }

  std::vector<Image8> rgb1 = ReadImageOrDie(argv[1]);
  std::vector<Image8> rgb2 = ReadImageOrDie(argv[2]);

  if (rgb1.size() != rgb2.size()) {
    fprintf(stderr, "Different number of channels: %lu vs %lu\n", rgb1.size(),
            rgb2.size());
    exit(1);
  }

  for (size_t c = 0; c < rgb1.size(); ++c) {
    if (rgb1[c].xsize() != rgb2[c].xsize() ||
        rgb1[c].ysize() != rgb2[c].ysize()) {
      fprintf(
          stderr, "The images are not equal in size: (%lu,%lu) vs (%lu,%lu)\n",
          rgb1[c].xsize(), rgb2[c].xsize(), rgb1[c].ysize(), rgb2[c].ysize());
      return 1;
    }
  }

  // TODO: Figure out if it is a good idea to fetch the gamma from the image
  // instead of applying sRGB conversion.
  std::vector<ImageF> linear1, linear2;
  // Overlay the image over a black background.
  FromSrgbToLinear(rgb1, linear1, 0);
  FromSrgbToLinear(rgb2, linear2, 0);
  ImageF diff_map, diff_map_on_white;
  double diff_value;
  if (!butteraugli::ButteraugliInterface(linear1, linear2, diff_map,
                                         diff_value)) {
    fprintf(stderr, "Butteraugli comparison failed\n");
    return 1;
  }
  ImageF* diff_map_ptr = &diff_map;
  if (rgb1.size() == 4 || rgb2.size() == 4) {
    // If the alpha channel is present, overlay the image over a white
    // background as well.
    FromSrgbToLinear(rgb1, linear1, 255);
    FromSrgbToLinear(rgb2, linear2, 255);
    double diff_value_on_white;
    if (!butteraugli::ButteraugliInterface(linear1, linear2, diff_map_on_white,
                                           diff_value_on_white)) {
      fprintf(stderr, "Butteraugli comparison failed\n");
      return 1;
    }
    if (diff_value_on_white > diff_value) {
      diff_value = diff_value_on_white;
      diff_map_ptr = &diff_map_on_white;
    }
  }
  printf("%lf\n", diff_value);

  if (argc == 4) {
    const double good_quality = ::butteraugli::ButteraugliFuzzyInverse(1.5);
    const double bad_quality = ::butteraugli::ButteraugliFuzzyInverse(0.5);
    std::vector<uint8_t> rgb;
    CreateHeatMapImage(*diff_map_ptr, good_quality, bad_quality,
                       rgb1[0].xsize(), rgb2[0].ysize(), &rgb);
    FILE* const fmap = fopen(argv[3], "wb");
    if (fmap == NULL) {
      fprintf(stderr, "Cannot open %s\n", argv[3]);
      perror("fopen");
      return 1;
    }
    bool ok = true;
    if (fprintf(fmap, "P6\n%lu %lu\n255\n",
                      rgb1[0].xsize(), rgb1[0].ysize()) < 0){
      perror("fprintf");
      ok = false;
    }
    if (ok && fwrite(rgb.data(), 1, rgb.size(), fmap) != rgb.size()) {
      perror("fwrite");
      ok = false;
    }
    if (fclose(fmap) != 0) {
      perror("fclose");
      ok = false;
    }
    if (!ok) return 1;
  }

  return 0;
}

}  // namespace
}  // namespace butteraugli

int main(int argc, char** argv) { return butteraugli::Run(argc, argv); }
