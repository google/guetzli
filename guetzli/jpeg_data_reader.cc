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

#include "guetzli/jpeg_data_reader.h"

#include <algorithm>
#include <stdio.h>
#include <string.h>

#include "guetzli/jpeg_huffman_decode.h"

namespace guetzli {

namespace {

// Macros for commonly used error conditions.

#define VERIFY_LEN(n)                                                   \
  if (*pos + (n) > len) {                                               \
    fprintf(stderr, "Unexpected end of input: pos=%d need=%d len=%d\n", \
            static_cast<int>(*pos), static_cast<int>(n),                \
            static_cast<int>(len));                                     \
    jpg->error = JPEG_UNEXPECTED_EOF;                                   \
    return false;                                                       \
  }

#define VERIFY_INPUT(var, low, high, code)                              \
  if (var < low || var > high) {                                        \
    fprintf(stderr, "Invalid %s: %d\n", #var, static_cast<int>(var));   \
    jpg->error = JPEG_INVALID_ ## code;                                 \
        return false;                                                   \
  }

#define VERIFY_MARKER_END()                                             \
  if (start_pos + marker_len != *pos) {                                 \
    fprintf(stderr, "Invalid marker length: declared=%d actual=%d\n",   \
            static_cast<int>(marker_len),                               \
            static_cast<int>(*pos - start_pos));                        \
    jpg->error = JPEG_WRONG_MARKER_SIZE;                                \
    return false;                                                       \
  }

#define EXPECT_MARKER() \
  if (pos + 2 > len || data[pos] != 0xff) {                             \
    fprintf(stderr, "Marker byte (0xff) expected, found: %d "           \
            "pos=%d len=%d\n",                                          \
            (pos < len ? data[pos] : 0), static_cast<int>(pos),         \
            static_cast<int>(len));                                     \
    jpg->error = JPEG_MARKER_BYTE_NOT_FOUND;                            \
    return false;                                                       \
  }

// Returns ceil(a/b).
inline int DivCeil(int a, int b) {
  return (a + b - 1) / b;
}

inline int ReadUint8(const uint8_t* data, size_t* pos) {
  return data[(*pos)++];
}

inline int ReadUint16(const uint8_t* data, size_t* pos) {
  int v = (data[*pos] << 8) + data[*pos + 1];
  *pos += 2;
  return v;
}

// Reads the Start of Frame (SOF) marker segment and fills in *jpg with the
// parsed data.
bool ProcessSOF(const uint8_t* data, const size_t len,
                JpegReadMode mode, size_t* pos, JPEGData* jpg) {
  if (jpg->width != 0) {
    fprintf(stderr, "Duplicate SOF marker.\n");
    jpg->error = JPEG_DUPLICATE_SOF;
    return false;
  }
  const size_t start_pos = *pos;
  VERIFY_LEN(8);
  size_t marker_len = ReadUint16(data, pos);
  int precision = ReadUint8(data, pos);
  int height = ReadUint16(data, pos);
  int width = ReadUint16(data, pos);
  int num_components = ReadUint8(data, pos);
  VERIFY_INPUT(precision, 8, 8, PRECISION);
  VERIFY_INPUT(height, 1, 65535, HEIGHT);
  VERIFY_INPUT(width, 1, 65535, WIDTH);
  VERIFY_INPUT(num_components, 1, kMaxComponents, NUMCOMP);
  VERIFY_LEN(3 * num_components);
  jpg->height = height;
  jpg->width = width;
  jpg->components.resize(num_components);

  // Read sampling factors and quant table index for each component.
  std::vector<bool> ids_seen(256, false);
  for (int i = 0; i < jpg->components.size(); ++i) {
    const int id = ReadUint8(data, pos);
    if (ids_seen[id]) {   // (cf. section B.2.2, syntax of Ci)
      fprintf(stderr, "Duplicate ID %d in SOF.\n", id);
      jpg->error = JPEG_DUPLICATE_COMPONENT_ID;
      return false;
    }
    ids_seen[id] = true;
    jpg->components[i].id = id;
    int factor = ReadUint8(data, pos);
    int h_samp_factor = factor >> 4;
    int v_samp_factor = factor & 0xf;
    VERIFY_INPUT(h_samp_factor, 1, 15, SAMP_FACTOR);
    VERIFY_INPUT(v_samp_factor, 1, 15, SAMP_FACTOR);
    jpg->components[i].h_samp_factor = h_samp_factor;
    jpg->components[i].v_samp_factor = v_samp_factor;
    jpg->components[i].quant_idx = ReadUint8(data, pos);
    jpg->max_h_samp_factor = std::max(jpg->max_h_samp_factor, h_samp_factor);
    jpg->max_v_samp_factor = std::max(jpg->max_v_samp_factor, v_samp_factor);
  }

  // We have checked above that none of the sampling factors are 0, so the max
  // sampling factors can not be 0.
  jpg->MCU_rows = DivCeil(jpg->height, jpg->max_v_samp_factor * 8);
  jpg->MCU_cols = DivCeil(jpg->width, jpg->max_h_samp_factor * 8);
  // Compute the block dimensions for each component.
  if (mode == JPEG_READ_ALL) {
    for (int i = 0; i < jpg->components.size(); ++i) {
      JPEGComponent* c = &jpg->components[i];
      if (jpg->max_h_samp_factor % c->h_samp_factor != 0 ||
          jpg->max_v_samp_factor % c->v_samp_factor != 0) {
        fprintf(stderr, "Non-integral subsampling ratios.\n");
        jpg->error = JPEG_INVALID_SAMPLING_FACTORS;
        return false;
      }
      c->width_in_blocks = jpg->MCU_cols * c->h_samp_factor;
      c->height_in_blocks = jpg->MCU_rows * c->v_samp_factor;
      const uint64_t num_blocks =
          static_cast<uint64_t>(c->width_in_blocks) * c->height_in_blocks;
      if (num_blocks > (1ull << 21)) {
        // Refuse to allocate more than 1 GB of memory for the coefficients,
        // that is 2M blocks x 64 coeffs x 2 bytes per coeff x max 4 components.
        // TODO(user) Add this limit to a GuetzliParams struct.
        fprintf(stderr, "Image too large.\n");
        jpg->error = JPEG_IMAGE_TOO_LARGE;
        return false;
      }
      c->num_blocks = static_cast<int>(num_blocks);
      c->coeffs.resize(c->num_blocks * kDCTBlockSize);
    }
  }
  VERIFY_MARKER_END();
  return true;
}

// Reads the Start of Scan (SOS) marker segment and fills in *scan_info with the
// parsed data.
bool ProcessSOS(const uint8_t* data, const size_t len, size_t* pos,
                JPEGData* jpg) {
  const size_t start_pos = *pos;
  VERIFY_LEN(3);
  size_t marker_len = ReadUint16(data, pos);
  int comps_in_scan = ReadUint8(data, pos);
  VERIFY_INPUT(comps_in_scan, 1, jpg->components.size(), COMPS_IN_SCAN);

  JPEGScanInfo scan_info;
  scan_info.components.resize(comps_in_scan);
  VERIFY_LEN(2 * comps_in_scan);
  std::vector<bool> ids_seen(256, false);
  for (int i = 0; i < comps_in_scan; ++i) {
    int id = ReadUint8(data, pos);
    if (ids_seen[id]) {   // (cf. section B.2.3, regarding CSj)
      fprintf(stderr, "Duplicate ID %d in SOS.\n", id);
      jpg->error = JPEG_DUPLICATE_COMPONENT_ID;
      return false;
    }
    ids_seen[id] = true;
    bool found_index = false;
    for (int j = 0; j < jpg->components.size(); ++j) {
      if (jpg->components[j].id == id) {
        scan_info.components[i].comp_idx = j;
        found_index = true;
      }
    }
    if (!found_index) {
      fprintf(stderr, "SOS marker: Could not find component with id %d\n", id);
      jpg->error = JPEG_COMPONENT_NOT_FOUND;
      return false;
    }
    int c = ReadUint8(data, pos);
    int dc_tbl_idx = c >> 4;
    int ac_tbl_idx = c & 0xf;
    VERIFY_INPUT(dc_tbl_idx, 0, 3, HUFFMAN_INDEX);
    VERIFY_INPUT(ac_tbl_idx, 0, 3, HUFFMAN_INDEX);
    scan_info.components[i].dc_tbl_idx = dc_tbl_idx;
    scan_info.components[i].ac_tbl_idx = ac_tbl_idx;
  }
  VERIFY_LEN(3);
  scan_info.Ss = ReadUint8(data, pos);
  scan_info.Se = ReadUint8(data, pos);
  VERIFY_INPUT(scan_info.Ss, 0, 63, START_OF_SCAN);
  VERIFY_INPUT(scan_info.Se, scan_info.Ss, 63, END_OF_SCAN);
  int c = ReadUint8(data, pos);
  scan_info.Ah = c >> 4;
  scan_info.Al = c & 0xf;
  // Check that all the Huffman tables needed for this scan are defined.
  for (int i = 0; i < comps_in_scan; ++i) {
    bool found_dc_table = false;
    bool found_ac_table = false;
    for (int j = 0; j < jpg->huffman_code.size(); ++j) {
      int slot_id = jpg->huffman_code[j].slot_id;
      if (slot_id == scan_info.components[i].dc_tbl_idx) {
        found_dc_table = true;
      } else if (slot_id == scan_info.components[i].ac_tbl_idx + 16) {
        found_ac_table = true;
      }
    }
    if (scan_info.Ss == 0 && !found_dc_table) {
      fprintf(stderr, "SOS marker: Could not find DC Huffman table with index "
              "%d\n", scan_info.components[i].dc_tbl_idx);
      jpg->error = JPEG_HUFFMAN_TABLE_NOT_FOUND;
      return false;
    }
    if (scan_info.Se > 0 && !found_ac_table) {
      fprintf(stderr, "SOS marker: Could not find AC Huffman table with index "
              "%d\n", scan_info.components[i].ac_tbl_idx);
      jpg->error = JPEG_HUFFMAN_TABLE_NOT_FOUND;
      return false;
    }
  }
  jpg->scan_info.push_back(scan_info);
  VERIFY_MARKER_END();
  return true;
}

// Reads the Define Huffman Table (DHT) marker segment and fills in *jpg with
// the parsed data. Builds the Huffman decoding table in either dc_huff_lut or
// ac_huff_lut, depending on the type and solt_id of Huffman code being read.
bool ProcessDHT(const uint8_t* data, const size_t len,
                JpegReadMode mode,
                std::vector<HuffmanTableEntry>* dc_huff_lut,
                std::vector<HuffmanTableEntry>* ac_huff_lut,
                size_t* pos,
                JPEGData* jpg) {
  const size_t start_pos = *pos;
  VERIFY_LEN(2);
  size_t marker_len = ReadUint16(data, pos);
  if (marker_len == 2) {
    fprintf(stderr, "DHT marker: no Huffman table found\n");
    jpg->error = JPEG_EMPTY_DHT;
    return false;
  }
  while (*pos < start_pos + marker_len) {
    VERIFY_LEN(1 + kJpegHuffmanMaxBitLength);
    JPEGHuffmanCode huff;
    huff.slot_id = ReadUint8(data, pos);
    int huffman_index = huff.slot_id;
    int is_ac_table = (huff.slot_id & 0x10) != 0;
    HuffmanTableEntry* huff_lut;
    if (is_ac_table) {
      huffman_index -= 0x10;
      VERIFY_INPUT(huffman_index, 0, 3, HUFFMAN_INDEX);
      huff_lut = &(*ac_huff_lut)[huffman_index * kJpegHuffmanLutSize];
    } else {
      VERIFY_INPUT(huffman_index, 0, 3, HUFFMAN_INDEX);
      huff_lut = &(*dc_huff_lut)[huffman_index * kJpegHuffmanLutSize];
    }
    huff.counts[0] = 0;
    int total_count = 0;
    int space = 1 << kJpegHuffmanMaxBitLength;
    int max_depth = 1;
    for (int i = 1; i <= kJpegHuffmanMaxBitLength; ++i) {
      int count = ReadUint8(data, pos);
      if (count != 0) {
        max_depth = i;
      }
      huff.counts[i] = count;
      total_count += count;
      space -= count * (1 << (kJpegHuffmanMaxBitLength - i));
    }
    if (is_ac_table) {
      VERIFY_INPUT(total_count, 0, kJpegHuffmanAlphabetSize, HUFFMAN_CODE);
    } else {
      VERIFY_INPUT(total_count, 0, kJpegDCAlphabetSize, HUFFMAN_CODE);
    }
    VERIFY_LEN(total_count);
    std::vector<bool> values_seen(256, false);
    for (int i = 0; i < total_count; ++i) {
      uint8_t value = ReadUint8(data, pos);
      if (!is_ac_table) {
        VERIFY_INPUT(value, 0, kJpegDCAlphabetSize - 1, HUFFMAN_CODE);
      }
      if (values_seen[value]) {
        fprintf(stderr, "Duplicate Huffman code value %d\n", value);
        jpg->error = JPEG_INVALID_HUFFMAN_CODE;
        return false;
      }
      values_seen[value] = true;
      huff.values[i] = value;
    }
    // Add an invalid symbol that will have the all 1 code.
    ++huff.counts[max_depth];
    huff.values[total_count] = kJpegHuffmanAlphabetSize;
    space -= (1 << (kJpegHuffmanMaxBitLength - max_depth));
    if (space < 0) {
      fprintf(stderr, "Invalid Huffman code lengths.\n");
      jpg->error = JPEG_INVALID_HUFFMAN_CODE;
      return false;
    } else if (space > 0 && huff_lut[0].value != 0xffff) {
      // Re-initialize the values to an invalid symbol so that we can recognize
      // it when reading the bit stream using a Huffman code with space > 0.
      for (int i = 0; i < kJpegHuffmanLutSize; ++i) {
        huff_lut[i].bits = 0;
        huff_lut[i].value = 0xffff;
      }
    }
    huff.is_last = (*pos == start_pos + marker_len);
    if (mode == JPEG_READ_ALL &&
        !BuildJpegHuffmanTable(&huff.counts[0], &huff.values[0], huff_lut)) {
      fprintf(stderr, "Failed to build Huffman table.\n");
      jpg->error = JPEG_INVALID_HUFFMAN_CODE;
      return false;
    }
    jpg->huffman_code.push_back(huff);
  }
  VERIFY_MARKER_END();
  return true;
}

// Reads the Define Quantization Table (DQT) marker segment and fills in *jpg
// with the parsed data.
bool ProcessDQT(const uint8_t* data, const size_t len, size_t* pos,
                JPEGData* jpg) {
  const size_t start_pos = *pos;
  VERIFY_LEN(2);
  size_t marker_len = ReadUint16(data, pos);
  if (marker_len == 2) {
    fprintf(stderr, "DQT marker: no quantization table found\n");
    jpg->error = JPEG_EMPTY_DQT;
    return false;
  }
  while (*pos < start_pos + marker_len && jpg->quant.size() < kMaxQuantTables) {
    VERIFY_LEN(1);
    int quant_table_index = ReadUint8(data, pos);
    int quant_table_precision = quant_table_index >> 4;
    quant_table_index &= 0xf;
    VERIFY_INPUT(quant_table_index, 0, 3, QUANT_TBL_INDEX);
    VERIFY_LEN((quant_table_precision ? 2 : 1) * kDCTBlockSize);
    JPEGQuantTable table;
    table.index = quant_table_index;
    table.precision = quant_table_precision;
    for (int i = 0; i < kDCTBlockSize; ++i) {
      int quant_val = quant_table_precision ?
          ReadUint16(data, pos) :
          ReadUint8(data, pos);
      VERIFY_INPUT(quant_val, 1, 65535, QUANT_VAL);
      table.values[kJPEGNaturalOrder[i]] = quant_val;
    }
    table.is_last = (*pos == start_pos + marker_len);
    jpg->quant.push_back(table);
  }
  VERIFY_MARKER_END();
  return true;
}

// Reads the DRI marker and saved the restart interval into *jpg.
bool ProcessDRI(const uint8_t* data, const size_t len, size_t* pos,
                JPEGData* jpg) {
  if (jpg->restart_interval > 0) {
    fprintf(stderr, "Duplicate DRI marker.\n");
    jpg->error = JPEG_DUPLICATE_DRI;
    return false;
  }
  const size_t start_pos = *pos;
  VERIFY_LEN(4);
  size_t marker_len = ReadUint16(data, pos);
  int restart_interval = ReadUint16(data, pos);
  jpg->restart_interval = restart_interval;
  VERIFY_MARKER_END();
  return true;
}

// Saves the APP marker segment as a string to *jpg.
bool ProcessAPP(const uint8_t* data, const size_t len, size_t* pos,
                JPEGData* jpg) {
  VERIFY_LEN(2);
  size_t marker_len = ReadUint16(data, pos);
  VERIFY_INPUT(marker_len, 2, 65535, MARKER_LEN);
  VERIFY_LEN(marker_len - 2);
  // Save the marker type together with the app data.
  std::string app_str(reinterpret_cast<const char*>(
      &data[*pos - 3]), marker_len + 1);
  *pos += marker_len - 2;
  jpg->app_data.push_back(app_str);
  return true;
}

// Saves the COM marker segment as a string to *jpg.
bool ProcessCOM(const uint8_t* data, const size_t len, size_t* pos,
                JPEGData* jpg) {
  VERIFY_LEN(2);
  size_t marker_len = ReadUint16(data, pos);
  VERIFY_INPUT(marker_len, 2, 65535, MARKER_LEN);
  VERIFY_LEN(marker_len - 2);
  std::string com_str(reinterpret_cast<const char*>(
      &data[*pos - 2]), marker_len);
  *pos += marker_len - 2;
  jpg->com_data.push_back(com_str);
  return true;
}

// Helper structure to read bits from the entropy coded data segment.
struct BitReaderState {
  BitReaderState(const uint8_t* data, const size_t len, size_t pos)
      : data_(data), len_(len) {
    Reset(pos);
  }

  void Reset(size_t pos) {
    pos_ = pos;
    val_ = 0;
    bits_left_ = 0;
    next_marker_pos_ = len_ - 2;
    FillBitWindow();
  }

  // Returns the next byte and skips the 0xff/0x00 escape sequences.
  uint8_t GetNextByte() {
    if (pos_ >= next_marker_pos_) {
      ++pos_;
      return 0;
    }
    uint8_t c = data_[pos_++];
    if (c == 0xff) {
      uint8_t escape = data_[pos_];
      if (escape == 0) {
        ++pos_;
      } else {
        // 0xff was followed by a non-zero byte, which means that we found the
        // start of the next marker segment.
        next_marker_pos_ = pos_ - 1;
      }
    }
    return c;
  }

  void FillBitWindow() {
    if (bits_left_ <= 16) {
      while (bits_left_ <= 56) {
        val_ <<= 8;
        val_ |= (uint64_t)GetNextByte();
        bits_left_ += 8;
      }
    }
  }

  int ReadBits(int nbits) {
    FillBitWindow();
    uint64_t val = (val_ >> (bits_left_ - nbits)) & ((1ULL << nbits) - 1);
    bits_left_ -= nbits;
    return val;
  }

  // Sets *pos to the next stream position where parsing should continue.
  // Returns false if the stream ended too early.
  bool FinishStream(size_t* pos) {
    // Give back some bytes that we did not use.
    int unused_bytes_left = bits_left_ >> 3;
    while (unused_bytes_left-- > 0) {
      --pos_;
      // If we give back a 0 byte, we need to check if it was a 0xff/0x00 escape
      // sequence, and if yes, we need to give back one more byte.
      if (pos_ < next_marker_pos_ &&
          data_[pos_] == 0 && data_[pos_ - 1] == 0xff) {
        --pos_;
      }
    }
    if (pos_ > next_marker_pos_) {
      // Data ran out before the scan was complete.
      fprintf(stderr, "Unexpected end of scan.\n");
      return false;
    }
    *pos = pos_;
    return true;
  }

  const uint8_t* data_;
  const size_t len_;
  size_t pos_;
  uint64_t val_;
  int bits_left_;
  size_t next_marker_pos_;
};

// Returns the next Huffman-coded symbol.
int ReadSymbol(const HuffmanTableEntry* table, BitReaderState* br) {
  int nbits;
  br->FillBitWindow();
  int val = (br->val_ >> (br->bits_left_ - 8)) & 0xff;
  table += val;
  nbits = table->bits - 8;
  if (nbits > 0) {
    br->bits_left_ -= 8;
    table += table->value;
    val = (br->val_ >> (br->bits_left_ - nbits)) & ((1 << nbits) - 1);
    table += val;
  }
  br->bits_left_ -= table->bits;
  return table->value;
}

// Returns the DC diff or AC value for extra bits value x and prefix code s.
// See Tables F.1 and F.2 of the spec.
int HuffExtend(int x, int s) {
  return (x < (1 << (s - 1)) ? x + ((-1) << s ) + 1 : x);
}

// Decodes one 8x8 block of DCT coefficients from the bit stream.
bool DecodeDCTBlock(const HuffmanTableEntry* dc_huff,
                    const HuffmanTableEntry* ac_huff,
                    int Ss, int Se, int Al,
                    int* eobrun,
                    BitReaderState* br,
                    JPEGData* jpg,
                    coeff_t* last_dc_coeff,
                    coeff_t* coeffs) {
  int s;
  int r;
  bool eobrun_allowed = Ss > 0;
  if (Ss == 0) {
    s = ReadSymbol(dc_huff, br);
    if (s >= kJpegDCAlphabetSize) {
      fprintf(stderr, "Invalid Huffman symbol %d for DC coefficient.\n", s);
      jpg->error = JPEG_INVALID_SYMBOL;
      return false;
    }
    if (s > 0) {
      r = br->ReadBits(s);
      s = HuffExtend(r, s);
    }
    s += *last_dc_coeff;
    const int dc_coeff = s << Al;
    coeffs[0] = dc_coeff;
    if (dc_coeff != coeffs[0]) {
      fprintf(stderr, "Invalid DC coefficient %d\n", dc_coeff);
      jpg->error = JPEG_NON_REPRESENTABLE_DC_COEFF;
      return false;
    }
    *last_dc_coeff = s;
    ++Ss;
  }
  if (Ss > Se) {
    return true;
  }
  if (*eobrun > 0) {
    --(*eobrun);
    return true;
  }
  for (int k = Ss; k <= Se; k++) {
    s = ReadSymbol(ac_huff, br);
    if (s >= kJpegHuffmanAlphabetSize) {
      fprintf(stderr, "Invalid Huffman symbol %d for AC coefficient %d\n",
              s, k);
      jpg->error = JPEG_INVALID_SYMBOL;
      return false;
    }
    r = s >> 4;
    s &= 15;
    if (s > 0) {
      k += r;
      if (k > Se) {
        fprintf(stderr, "Out-of-band coefficient %d band was %d-%d\n",
                k, Ss, Se);
        jpg->error = JPEG_OUT_OF_BAND_COEFF;
        return false;
      }
      if (s + Al >= kJpegDCAlphabetSize) {
        fprintf(stderr, "Out of range AC coefficient value: s=%d Al=%d k=%d\n",
                s, Al, k);
        jpg->error = JPEG_NON_REPRESENTABLE_AC_COEFF;
        return false;
      }
      r = br->ReadBits(s);
      s = HuffExtend(r, s);
      coeffs[kJPEGNaturalOrder[k]] = s << Al;
    } else if (r == 15) {
      k += 15;
    } else {
      *eobrun = 1 << r;
      if (r > 0) {
        if (!eobrun_allowed) {
          fprintf(stderr, "End-of-block run crossing DC coeff.\n");
          jpg->error = JPEG_EOB_RUN_TOO_LONG;
          return false;
        }
        *eobrun += br->ReadBits(r);
      }
      break;
    }
  }
  --(*eobrun);
  return true;
}

bool RefineDCTBlock(const HuffmanTableEntry* ac_huff,
                    int Ss, int Se, int Al,
                    int* eobrun,
                    BitReaderState* br,
                    JPEGData* jpg,
                    coeff_t* coeffs) {
  bool eobrun_allowed = Ss > 0;
  if (Ss == 0) {
    int s = br->ReadBits(1);
    coeff_t dc_coeff = coeffs[0];
    dc_coeff |= s << Al;
    coeffs[0] = dc_coeff;
    ++Ss;
  }
  if (Ss > Se) {
    return true;
  }
  int p1 = 1 << Al;
  int m1 = (-1) << Al;
  int k = Ss;
  int r;
  int s;
  bool in_zero_run = false;
  if (*eobrun <= 0) {
    for (; k <= Se; k++) {
      s = ReadSymbol(ac_huff, br);
      if (s >= kJpegHuffmanAlphabetSize) {
        fprintf(stderr, "Invalid Huffman symbol %d for AC coefficient %d\n",
                s, k);
        jpg->error = JPEG_INVALID_SYMBOL;
        return false;
      }
      r = s >> 4;
      s &= 15;
      if (s) {
        if (s != 1) {
          fprintf(stderr, "Invalid Huffman symbol %d for AC coefficient %d\n",
                  s, k);
          jpg->error = JPEG_INVALID_SYMBOL;
          return false;
        }
        s = br->ReadBits(1) ? p1 : m1;
        in_zero_run = false;
      } else {
        if (r != 15) {
          *eobrun = 1 << r;
          if (r > 0) {
            if (!eobrun_allowed) {
              fprintf(stderr, "End-of-block run crossing DC coeff.\n");
              jpg->error = JPEG_EOB_RUN_TOO_LONG;
              return false;
            }
            *eobrun += br->ReadBits(r);
          }
          break;
        }
        in_zero_run = true;
      }
      do {
        coeff_t thiscoef = coeffs[kJPEGNaturalOrder[k]];
        if (thiscoef != 0) {
          if (br->ReadBits(1)) {
            if ((thiscoef & p1) == 0) {
              if (thiscoef >= 0) {
                thiscoef += p1;
              } else {
                thiscoef += m1;
              }
            }
          }
          coeffs[kJPEGNaturalOrder[k]] = thiscoef;
        } else {
          if (--r < 0) {
            break;
          }
        }
        k++;
      } while (k <= Se);
      if (s) {
        if (k > Se) {
          fprintf(stderr, "Out-of-band coefficient %d band was %d-%d\n",
                  k, Ss, Se);
          jpg->error = JPEG_OUT_OF_BAND_COEFF;
          return false;
        }
        coeffs[kJPEGNaturalOrder[k]] = s;
      }
    }
  }
  if (in_zero_run) {
    fprintf(stderr, "Extra zero run before end-of-block.\n");
    jpg->error = JPEG_EXTRA_ZERO_RUN;
    return false;
  }
  if (*eobrun > 0) {
    for (; k <= Se; k++) {
      coeff_t thiscoef = coeffs[kJPEGNaturalOrder[k]];
      if (thiscoef != 0) {
        if (br->ReadBits(1)) {
          if ((thiscoef & p1) == 0) {
            if (thiscoef >= 0) {
              thiscoef += p1;
            } else {
              thiscoef += m1;
            }
          }
        }
        coeffs[kJPEGNaturalOrder[k]] = thiscoef;
      }
    }
  }
  --(*eobrun);
  return true;
}

bool ProcessRestart(const uint8_t* data, const size_t len,
                    int* next_restart_marker, BitReaderState* br,
                    JPEGData* jpg) {
  size_t pos = 0;
  if (!br->FinishStream(&pos)) {
    jpg->error = JPEG_INVALID_SCAN;
    return false;
  }
  int expected_marker = 0xd0 + *next_restart_marker;
  EXPECT_MARKER();
  int marker = data[pos + 1];
  if (marker != expected_marker) {
    fprintf(stderr, "Did not find expected restart marker %d actual=%d\n",
            expected_marker, marker);
    jpg->error = JPEG_WRONG_RESTART_MARKER;
    return false;
  }
  br->Reset(pos + 2);
  *next_restart_marker += 1;
  *next_restart_marker &= 0x7;
  return true;
}

bool ProcessScan(const uint8_t* data, const size_t len,
                 const std::vector<HuffmanTableEntry>& dc_huff_lut,
                 const std::vector<HuffmanTableEntry>& ac_huff_lut,
                 uint16_t scan_progression[kMaxComponents][kDCTBlockSize],
                 bool is_progressive,
                 size_t* pos,
                 JPEGData* jpg) {
  if (!ProcessSOS(data, len, pos, jpg)) {
    return false;
  }
  JPEGScanInfo* scan_info = &jpg->scan_info.back();
  bool is_interleaved = (scan_info->components.size() > 1);
  int MCUs_per_row;
  int MCU_rows;
  if (is_interleaved) {
    MCUs_per_row = jpg->MCU_cols;
    MCU_rows = jpg->MCU_rows;
  } else {
    const JPEGComponent& c = jpg->components[scan_info->components[0].comp_idx];
    MCUs_per_row =
        DivCeil(jpg->width * c.h_samp_factor, 8 * jpg->max_h_samp_factor);
    MCU_rows =
        DivCeil(jpg->height * c.v_samp_factor, 8 * jpg->max_v_samp_factor);
  }
  coeff_t last_dc_coeff[kMaxComponents] = {0};
  BitReaderState br(data, len, *pos);
  int restarts_to_go = jpg->restart_interval;
  int next_restart_marker = 0;
  int eobrun = -1;
  int block_scan_index = 0;
  const int Al = is_progressive ? scan_info->Al : 0;
  const int Ah = is_progressive ? scan_info->Ah : 0;
  const int Ss = is_progressive ? scan_info->Ss : 0;
  const int Se = is_progressive ? scan_info->Se : 63;
  const uint16_t scan_bitmask = Ah == 0 ? (0xffff << Al) : (1u << Al);
  const uint16_t refinement_bitmask = (1 << Al) - 1;
  for (int i = 0; i < scan_info->components.size(); ++i) {
    int comp_idx = scan_info->components[i].comp_idx;
    for (int k = Ss; k <= Se; ++k) {
      if (scan_progression[comp_idx][k] & scan_bitmask) {
        fprintf(stderr, "Overlapping scans: component=%d k=%d prev_mask=%d "
                "cur_mask=%d\n", comp_idx, k, scan_progression[i][k],
                scan_bitmask);
        jpg->error = JPEG_OVERLAPPING_SCANS;
        return false;
      }
      if (scan_progression[comp_idx][k] & refinement_bitmask) {
        fprintf(stderr, "Invalid scan order, a more refined scan was already "
                "done: component=%d k=%d prev_mask=%d cur_mask=%d\n", comp_idx,
                k, scan_progression[i][k], scan_bitmask);
        jpg->error = JPEG_INVALID_SCAN_ORDER;
        return false;
      }
      scan_progression[comp_idx][k] |= scan_bitmask;
    }
  }
  if (Al > 10) {
    fprintf(stderr, "Scan parameter Al=%d is not supported in guetzli.\n", Al);
    jpg->error = JPEG_NON_REPRESENTABLE_AC_COEFF;
    return false;
  }
  for (int mcu_y = 0; mcu_y < MCU_rows; ++mcu_y) {
    for (int mcu_x = 0; mcu_x < MCUs_per_row; ++mcu_x) {
      // Handle the restart intervals.
      if (jpg->restart_interval > 0) {
        if (restarts_to_go == 0) {
          if (ProcessRestart(data, len,
                             &next_restart_marker, &br, jpg)) {
            restarts_to_go = jpg->restart_interval;
            memset(last_dc_coeff, 0, sizeof(last_dc_coeff));
            if (eobrun > 0) {
              fprintf(stderr, "End-of-block run too long.\n");
              jpg->error = JPEG_EOB_RUN_TOO_LONG;
              return false;
            }
            eobrun = -1;   // fresh start
          } else {
            return false;
          }
        }
        --restarts_to_go;
      }
      // Decode one MCU.
      for (int i = 0; i < scan_info->components.size(); ++i) {
        JPEGComponentScanInfo* si = &scan_info->components[i];
        JPEGComponent* c = &jpg->components[si->comp_idx];
        const HuffmanTableEntry* dc_lut =
            &dc_huff_lut[si->dc_tbl_idx * kJpegHuffmanLutSize];
        const HuffmanTableEntry* ac_lut =
            &ac_huff_lut[si->ac_tbl_idx * kJpegHuffmanLutSize];
        int nblocks_y = is_interleaved ? c->v_samp_factor : 1;
        int nblocks_x = is_interleaved ? c->h_samp_factor : 1;
        for (int iy = 0; iy < nblocks_y; ++iy) {
          for (int ix = 0; ix < nblocks_x; ++ix) {
            int block_y = mcu_y * nblocks_y + iy;
            int block_x = mcu_x * nblocks_x + ix;
            int block_idx = block_y * c->width_in_blocks + block_x;
            coeff_t* coeffs = &c->coeffs[block_idx * kDCTBlockSize];
            if (Ah == 0) {
              if (!DecodeDCTBlock(dc_lut, ac_lut, Ss, Se, Al, &eobrun, &br, jpg,
                                  &last_dc_coeff[si->comp_idx], coeffs)) {
                return false;
              }
            } else {
              if (!RefineDCTBlock(ac_lut, Ss, Se, Al,
                                  &eobrun, &br, jpg, coeffs)) {
                return false;
              }
            }
            ++block_scan_index;
          }
        }
      }
    }
  }
  if (eobrun > 0) {
    fprintf(stderr, "End-of-block run too long.\n");
    jpg->error = JPEG_EOB_RUN_TOO_LONG;
    return false;
  }
  if (!br.FinishStream(pos)) {
    jpg->error = JPEG_INVALID_SCAN;
    return false;
  }
  if (*pos > len) {
    fprintf(stderr, "Unexpected end of file during scan. pos=%d len=%d\n",
            static_cast<int>(*pos), static_cast<int>(len));
    jpg->error = JPEG_UNEXPECTED_EOF;
    return false;
  }
  return true;
}

// Changes the quant_idx field of the components to refer to the index of the
// quant table in the jpg->quant array.
bool FixupIndexes(JPEGData* jpg) {
  for (int i = 0; i < jpg->components.size(); ++i) {
    JPEGComponent* c = &jpg->components[i];
    bool found_index = false;
    for (int j = 0; j < jpg->quant.size(); ++j) {
      if (jpg->quant[j].index == c->quant_idx) {
        c->quant_idx = j;
        found_index = true;
        break;
      }
    }
    if (!found_index) {
      fprintf(stderr, "Quantization table with index %d not found\n",
              c->quant_idx);
      jpg->error = JPEG_QUANT_TABLE_NOT_FOUND;
      return false;
    }
  }
  return true;
}

size_t FindNextMarker(const uint8_t* data, const size_t len, size_t pos) {
  // kIsValidMarker[i] == 1 means (0xc0 + i) is a valid marker.
  static const uint8_t kIsValidMarker[] = {
    1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
  };
  size_t num_skipped = 0;
  while (pos + 1 < len &&
         (data[pos] != 0xff || data[pos + 1] < 0xc0 ||
          !kIsValidMarker[data[pos + 1] - 0xc0])) {
    ++pos;
    ++num_skipped;
  }
  return num_skipped;
}

}  // namespace

bool ReadJpeg(const uint8_t* data, const size_t len, JpegReadMode mode,
              JPEGData* jpg) {
  size_t pos = 0;
  // Check SOI marker.
  EXPECT_MARKER();
  int marker = data[pos + 1];
  pos += 2;
  if (marker != 0xd8) {
    fprintf(stderr, "Did not find expected SOI marker, actual=%d\n", marker);
    jpg->error = JPEG_SOI_NOT_FOUND;
    return false;
  }
  int lut_size = kMaxHuffmanTables * kJpegHuffmanLutSize;
  std::vector<HuffmanTableEntry> dc_huff_lut(lut_size);
  std::vector<HuffmanTableEntry> ac_huff_lut(lut_size);
  bool found_sof = false;
  uint16_t scan_progression[kMaxComponents][kDCTBlockSize] = { { 0 } };

  bool is_progressive = false;   // default
  do {
    // Read next marker.
    size_t num_skipped = FindNextMarker(data, len, pos);
    if (num_skipped > 0) {
      // Add a fake marker to indicate arbitrary in-between-markers data.
      jpg->marker_order.push_back(0xff);
      jpg->inter_marker_data.push_back(
          std::string(reinterpret_cast<const char*>(&data[pos]),
                                      num_skipped));
      pos += num_skipped;
    }
    EXPECT_MARKER();
    marker = data[pos + 1];
    pos += 2;
    bool ok = true;
    switch (marker) {
      case 0xc0:
      case 0xc1:
      case 0xc2:
        is_progressive = (marker == 0xc2);
        ok = ProcessSOF(data, len, mode, &pos, jpg);
        found_sof = true;
        break;
      case 0xc4:
        ok = ProcessDHT(data, len, mode, &dc_huff_lut, &ac_huff_lut, &pos, jpg);
        break;
      case 0xd0:
      case 0xd1:
      case 0xd2:
      case 0xd3:
      case 0xd4:
      case 0xd5:
      case 0xd6:
      case 0xd7:
        // RST markers do not have any data.
        break;
      case 0xd9:
        // Found end marker.
        break;
      case 0xda:
        if (mode == JPEG_READ_ALL) {
          ok = ProcessScan(data, len, dc_huff_lut, ac_huff_lut,
                           scan_progression, is_progressive, &pos, jpg);
        }
        break;
      case 0xdb:
        ok = ProcessDQT(data, len, &pos, jpg);
        break;
      case 0xdd:
        ok = ProcessDRI(data, len, &pos, jpg);
        break;
      case 0xe0:
      case 0xe1:
      case 0xe2:
      case 0xe3:
      case 0xe4:
      case 0xe5:
      case 0xe6:
      case 0xe7:
      case 0xe8:
      case 0xe9:
      case 0xea:
      case 0xeb:
      case 0xec:
      case 0xed:
      case 0xee:
      case 0xef:
        if (mode != JPEG_READ_TABLES) {
          ok = ProcessAPP(data, len, &pos, jpg);
        }
        break;
      case 0xfe:
        if (mode != JPEG_READ_TABLES) {
          ok = ProcessCOM(data, len, &pos, jpg);
        }
        break;
      default:
        fprintf(stderr, "Unsupported marker: %d pos=%d len=%d\n",
                marker, static_cast<int>(pos), static_cast<int>(len));
        jpg->error = JPEG_UNSUPPORTED_MARKER;
        ok = false;
        break;
    }
    if (!ok) {
      return false;
    }
    jpg->marker_order.push_back(marker);
    if (mode == JPEG_READ_HEADER && found_sof) {
      break;
    }
  } while (marker != 0xd9);

  if (!found_sof) {
    fprintf(stderr, "Missing SOF marker.\n");
    jpg->error = JPEG_SOF_NOT_FOUND;
    return false;
  }

  // Supplemental checks.
  if (mode == JPEG_READ_ALL) {
    if (pos < len) {
      jpg->tail_data.assign(reinterpret_cast<const char*>(&data[pos]),
                            len - pos);
    }
    if (!FixupIndexes(jpg)) {
      return false;
    }
    if (jpg->huffman_code.size() == 0) {
      // Section B.2.4.2: "If a table has never been defined for a particular
      // destination, then when this destination is specified in a scan header,
      // the results are unpredictable."
      fprintf(stderr, "Need at least one Huffman code table.\n");
      jpg->error = JPEG_HUFFMAN_TABLE_ERROR;
      return false;
    }
    if (jpg->huffman_code.size() >= kMaxDHTMarkers) {
      fprintf(stderr, "Too many Huffman tables.\n");
      jpg->error = JPEG_HUFFMAN_TABLE_ERROR;
      return false;
    }
  }
  return true;
}

bool ReadJpeg(const std::string& data, JpegReadMode mode,
              JPEGData* jpg) {
  return ReadJpeg(reinterpret_cast<const uint8_t*>(data.data()),
                  static_cast<const size_t>(data.size()),
                  mode, jpg);
}

}  // namespace guetzli
