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

#include "guetzli/jpeg_data_writer.h"

#include <assert.h>
#include <cstdlib>
#include <string.h>

#include "guetzli/entropy_encode.h"
#include "guetzli/fast_log.h"
#include "guetzli/jpeg_bit_writer.h"

namespace guetzli {

namespace {

static const int kJpegPrecision = 8;

// Writes len bytes from buf, using the out callback.
inline bool JPEGWrite(JPEGOutput out, const uint8_t* buf, size_t len) {
  static const size_t kBlockSize = 1u << 30;
  size_t pos = 0;
  while (len - pos > kBlockSize) {
    if (!out.Write(buf + pos, kBlockSize)) {
      return false;
    }
    pos += kBlockSize;
  }
  return out.Write(buf + pos, len - pos);
}

// Writes a string using the out callback.
inline bool JPEGWrite(JPEGOutput out, const std::string& s) {
  const uint8_t* data = reinterpret_cast<const uint8_t*>(&s[0]);
  return JPEGWrite(out, data, s.size());
}

bool EncodeMetadata(const JPEGData& jpg, bool strip_metadata, JPEGOutput out) {
  if (strip_metadata) {
    const uint8_t kApp0Data[] = {
      0xff, 0xe0, 0x00, 0x10,        // APP0
      0x4a, 0x46, 0x49, 0x46, 0x00,  // 'JFIF'
      0x01, 0x01,                    // v1.01
      0x00, 0x00, 0x01, 0x00, 0x01,  // aspect ratio = 1:1
      0x00, 0x00                     // thumbnail width/height
    };
    return JPEGWrite(out, kApp0Data, sizeof(kApp0Data));
  }
  bool ok = true;
  for (int i = 0; i < jpg.app_data.size(); ++i) {
    uint8_t data[1] = { 0xff };
    ok = ok && JPEGWrite(out, data, sizeof(data));
    ok = ok && JPEGWrite(out, jpg.app_data[i]);
  }
  for (int i = 0; i < jpg.com_data.size(); ++i) {
    uint8_t data[2] = { 0xff, 0xfe };
    ok = ok && JPEGWrite(out, data, sizeof(data));
    ok = ok && JPEGWrite(out, jpg.com_data[i]);
  }
  return ok;
}

bool EncodeDQT(const std::vector<JPEGQuantTable>& quant, JPEGOutput out) {
  int marker_len = 2;
  for (int i = 0; i < quant.size(); ++i) {
    marker_len += 1 + (quant[i].precision ? 2 : 1) * kDCTBlockSize;
  }
  std::vector<uint8_t> data(marker_len + 2);
  size_t pos = 0;
  data[pos++] = 0xff;
  data[pos++] = 0xdb;
  data[pos++] = marker_len >> 8;
  data[pos++] = marker_len & 0xff;
  for (int i = 0; i < quant.size(); ++i) {
    const JPEGQuantTable& table = quant[i];
    data[pos++] = (table.precision << 4) + table.index;
    for (int k = 0; k < kDCTBlockSize; ++k) {
      int val = table.values[kJPEGNaturalOrder[k]];
      if (table.precision) {
        data[pos++] = val >> 8;
      }
      data[pos++] = val & 0xff;
    }
  }
  return JPEGWrite(out, &data[0], pos);
}

bool EncodeSOF(const JPEGData& jpg, JPEGOutput out) {
  const size_t ncomps = jpg.components.size();
  const size_t marker_len = 8 + 3 * ncomps;
  std::vector<uint8_t> data(marker_len + 2);
  size_t pos = 0;
  data[pos++] = 0xff;
  data[pos++] = 0xc1;
  data[pos++] = marker_len >> 8;
  data[pos++] = marker_len & 0xff;
  data[pos++] = kJpegPrecision;
  data[pos++] = jpg.height >> 8;
  data[pos++] = jpg.height & 0xff;
  data[pos++] = jpg.width >> 8;
  data[pos++] = jpg.width & 0xff;
  data[pos++] = ncomps;
  for (size_t i = 0; i < ncomps; ++i) {
    data[pos++] = jpg.components[i].id;
    data[pos++] = ((jpg.components[i].h_samp_factor << 4) |
                      (jpg.components[i].v_samp_factor));
    const int quant_idx = jpg.components[i].quant_idx;
    if (quant_idx >= jpg.quant.size()) {
      return false;
    }
    data[pos++] = jpg.quant[quant_idx].index;
  }
  return JPEGWrite(out, &data[0], pos);
}

// Builds a JPEG-style huffman code from the given bit depths.
void BuildHuffmanCode(uint8_t* depth, int* counts, int* values) {
  for (int i = 0; i < JpegHistogram::kSize; ++i) {
    if (depth[i] > 0) {
      ++counts[depth[i]];
    }
  }
  int offset[kJpegHuffmanMaxBitLength + 1] = { 0 };
  for (int i = 1; i <= kJpegHuffmanMaxBitLength; ++i) {
    offset[i] = offset[i - 1] + counts[i - 1];
  }
  for (int i = 0; i < JpegHistogram::kSize; ++i) {
    if (depth[i] > 0) {
      values[offset[depth[i]]++] = i;
    }
  }
}

void BuildHuffmanCodeTable(const int* counts, const int* values,
                           HuffmanCodeTable* table) {
  int huffcode[256];
  int huffsize[256];
  int p = 0;
  for (int l = 1; l <= kJpegHuffmanMaxBitLength; ++l) {
    int i = counts[l];
    while (i--) huffsize[p++] = l;
  }

  if (p == 0)
    return;

  huffsize[p - 1] = 0;
  int lastp = p - 1;

  int code = 0;
  int si = huffsize[0];
  p = 0;
  while (huffsize[p]) {
    while ((huffsize[p]) == si) {
      huffcode[p++] = code;
      code++;
    }
    code <<= 1;
    si++;
  }
  for (p = 0; p < lastp; p++) {
    int i = values[p];
    table->depth[i] = huffsize[p];
    table->code[i] = huffcode[p];
  }
}

}  // namespace

// Updates ac_histogram with the counts of the AC symbols that will be added by
// a sequential jpeg encoder for this block. Every symbol is counted twice so
// that we can add a fake symbol at the end with count 1 to be the last (least
// frequent) symbol with the all 1 code.
void UpdateACHistogramForDCTBlock(const coeff_t* coeffs,
                                  JpegHistogram* ac_histogram) {
  int r = 0;
  for (int k = 1; k < 64; ++k) {
    coeff_t coeff = coeffs[kJPEGNaturalOrder[k]];
    if (coeff == 0) {
      r++;
      continue;
    }
    while (r > 15) {
      ac_histogram->Add(0xf0);
      r -= 16;
    }
    int nbits = Log2FloorNonZero(std::abs(coeff)) + 1;
    int symbol = (r << 4) + nbits;
    ac_histogram->Add(symbol);
    r = 0;
  }
  if (r > 0) {
    ac_histogram->Add(0);
  }
}

size_t HistogramHeaderCost(const JpegHistogram& histo) {
  size_t header_bits = 17 * 8;
  for (int i = 0; i + 1 < JpegHistogram::kSize; ++i) {
    if (histo.counts[i] > 0) {
      header_bits += 8;
    }
  }
  return header_bits;
}

size_t HistogramEntropyCost(const JpegHistogram& histo,
                            const uint8_t depths[256]) {
  size_t bits = 0;
  for (int i = 0; i + 1 < JpegHistogram::kSize; ++i) {
    // JpegHistogram::Add() counts every symbol twice, so we have to divide by
    // two here.
    bits += (histo.counts[i] / 2) * (depths[i] + (i & 0xf));
  }
  // Estimate escape byte rate to be 0.75/256.
  bits += (bits * 3 + 512) >> 10;
  return bits;
}

void BuildDCHistograms(const JPEGData& jpg, JpegHistogram* histo) {
  for (int i = 0; i < jpg.components.size(); ++i) {
    const JPEGComponent& c = jpg.components[i];
    JpegHistogram* dc_histogram = &histo[i];
    coeff_t last_dc_coeff = 0;
    for (int mcu_y = 0; mcu_y < jpg.MCU_rows; ++mcu_y) {
      for (int mcu_x = 0; mcu_x < jpg.MCU_cols; ++mcu_x) {
        for (int iy = 0; iy < c.v_samp_factor; ++iy) {
          for (int ix = 0; ix < c.h_samp_factor; ++ix) {
            int block_y = mcu_y * c.v_samp_factor + iy;
            int block_x = mcu_x * c.h_samp_factor + ix;
            int block_idx = block_y * c.width_in_blocks + block_x;
            coeff_t dc_coeff = c.coeffs[block_idx << 6];
            int diff = std::abs(dc_coeff - last_dc_coeff);
            int nbits = Log2Floor(diff) + 1;
            dc_histogram->Add(nbits);
            last_dc_coeff = dc_coeff;
          }
        }
      }
    }
  }
}

void BuildACHistograms(const JPEGData& jpg, JpegHistogram* histo) {
  for (int i = 0; i < jpg.components.size(); ++i) {
    const JPEGComponent& c = jpg.components[i];
    JpegHistogram* ac_histogram = &histo[i];
    for (int j = 0; j < c.coeffs.size(); j += kDCTBlockSize) {
      UpdateACHistogramForDCTBlock(&c.coeffs[j], ac_histogram);
    }
  }
}

// Size of everything except the Huffman codes and the entropy coded data.
size_t JpegHeaderSize(const JPEGData& jpg, bool strip_metadata) {
  size_t num_bytes = 0;
  num_bytes += 2;  // SOI
  if (strip_metadata) {
    num_bytes += 18;  // APP0
  } else {
    for (int i = 0; i < jpg.app_data.size(); ++i) {
      num_bytes += 1 + jpg.app_data[i].size();
    }
    for (int i = 0; i < jpg.com_data.size(); ++i) {
      num_bytes += 2 + jpg.com_data[i].size();
    }
  }
  // DQT
  num_bytes += 4;
  for (int i = 0; i < jpg.quant.size(); ++i) {
    num_bytes += 1 + (jpg.quant[i].precision ? 2 : 1) * kDCTBlockSize;
  }
  num_bytes += 10 + 3 * jpg.components.size();  // SOF
  num_bytes += 4;  // DHT (w/o actual Huffman code data)
  num_bytes += 8 + 2 * jpg.components.size();  // SOS
  num_bytes += 2;  // EOI
  num_bytes += jpg.tail_data.size();
  return num_bytes;
}

size_t ClusterHistograms(JpegHistogram* histo, size_t* num,
                         int* histo_indexes, uint8_t* depth) {
  memset(depth, 0, *num * JpegHistogram::kSize);
  size_t costs[kMaxComponents];
  for (size_t i = 0; i < *num; ++i) {
    histo_indexes[i] = i;
    std::vector<HuffmanTree> tree(2 * JpegHistogram::kSize + 1);
    CreateHuffmanTree(histo[i].counts, JpegHistogram::kSize,
                      kJpegHuffmanMaxBitLength, &tree[0],
                      &depth[i * JpegHistogram::kSize]);
    costs[i] = (HistogramHeaderCost(histo[i]) +
                HistogramEntropyCost(histo[i],
                                     &depth[i * JpegHistogram::kSize]));
  }
  const size_t orig_num = *num;
  while (*num > 1) {
    size_t last = *num - 1;
    size_t second_last = *num - 2;
    JpegHistogram combined(histo[last]);
    combined.AddHistogram(histo[second_last]);
    std::vector<HuffmanTree> tree(2 * JpegHistogram::kSize + 1);
    uint8_t depth_combined[JpegHistogram::kSize] = { 0 };
    CreateHuffmanTree(combined.counts, JpegHistogram::kSize,
                      kJpegHuffmanMaxBitLength, &tree[0], depth_combined);
    size_t cost_combined = (HistogramHeaderCost(combined) +
                            HistogramEntropyCost(combined, depth_combined));
    if (cost_combined < costs[last] + costs[second_last]) {
      histo[second_last] = combined;
      histo[last] = JpegHistogram();
      costs[second_last] = cost_combined;
      memcpy(&depth[second_last * JpegHistogram::kSize], depth_combined,
             sizeof(depth_combined));
      for (size_t i = 0; i < orig_num; ++i) {
        if (histo_indexes[i] == last) {
          histo_indexes[i] = second_last;
        }
      }
      --(*num);
    } else {
      break;
    }
  }
  size_t total_cost = 0;
  for (int i = 0; i < *num; ++i) {
    total_cost += costs[i];
  }
  return (total_cost + 7) / 8;
}

size_t EstimateJpegDataSize(const int num_components,
                            const std::vector<JpegHistogram>& histograms) {
  assert(histograms.size() == 2 * num_components);
  std::vector<JpegHistogram> clustered = histograms;
  size_t num_dc = num_components;
  size_t num_ac = num_components;
  int indexes[kMaxComponents];
  uint8_t depth[kMaxComponents * JpegHistogram::kSize];
  return (ClusterHistograms(&clustered[0], &num_dc, indexes, depth) +
          ClusterHistograms(&clustered[num_components], &num_ac, indexes,
                            depth));
}

namespace {

// Writes DHT and SOS marker segments to out and fills in DC/AC Huffman tables
// for each component of the image.
bool BuildAndEncodeHuffmanCodes(const JPEGData& jpg, JPEGOutput out,
                                std::vector<HuffmanCodeTable>* dc_huff_tables,
                                std::vector<HuffmanCodeTable>* ac_huff_tables) {
  const int ncomps = jpg.components.size();
  dc_huff_tables->resize(ncomps);
  ac_huff_tables->resize(ncomps);

  // Build separate DC histograms for each component.
  std::vector<JpegHistogram> histograms(ncomps);
  BuildDCHistograms(jpg, &histograms[0]);

  // Cluster DC histograms.
  size_t num_dc_histo = ncomps;
  int dc_histo_indexes[kMaxComponents];
  std::vector<uint8_t> depths(ncomps * JpegHistogram::kSize);
  ClusterHistograms(&histograms[0], &num_dc_histo, dc_histo_indexes,
                    &depths[0]);

  // Build separate AC histograms for each component.
  histograms.resize(num_dc_histo + ncomps);
  depths.resize((num_dc_histo + ncomps) * JpegHistogram::kSize);
  BuildACHistograms(jpg, &histograms[num_dc_histo]);

  // Cluster AC histograms.
  size_t num_ac_histo = ncomps;
  int ac_histo_indexes[kMaxComponents];
  ClusterHistograms(&histograms[num_dc_histo], &num_ac_histo, ac_histo_indexes,
                    &depths[num_dc_histo * JpegHistogram::kSize]);

  // Compute DHT and SOS marker data sizes and start emitting DHT marker.
  int num_histo = num_dc_histo + num_ac_histo;
  histograms.resize(num_histo);
  int total_count = 0;
  for (int i = 0; i < histograms.size(); ++i) {
    total_count += histograms[i].NumSymbols();
  }
  const size_t dht_marker_len =
      2 + num_histo * (kJpegHuffmanMaxBitLength + 1) + total_count;
  const size_t sos_marker_len = 6 + 2 * ncomps;
  std::vector<uint8_t> data(dht_marker_len + sos_marker_len + 4);
  size_t pos = 0;
  data[pos++] = 0xff;
  data[pos++] = 0xc4;
  data[pos++] = dht_marker_len >> 8;
  data[pos++] = dht_marker_len & 0xff;

  // Compute Huffman codes for each histograms.
  for (size_t i = 0; i < num_histo; ++i) {
    const bool is_dc = i < num_dc_histo;
    const int idx = is_dc ? i : i - num_dc_histo;
    int counts[kJpegHuffmanMaxBitLength + 1] = { 0 };
    int values[JpegHistogram::kSize] = { 0 };
    BuildHuffmanCode(&depths[i * JpegHistogram::kSize], counts, values);
    HuffmanCodeTable table;
    for (int j = 0; j < 256; ++j) table.depth[j] = 255;
    BuildHuffmanCodeTable(counts, values, &table);
    for (int c = 0; c < ncomps; ++c) {
      if (is_dc) {
        if (dc_histo_indexes[c] == idx) (*dc_huff_tables)[c] = table;
      } else {
        if (ac_histo_indexes[c] == idx) (*ac_huff_tables)[c] = table;
      }
    }
    int max_length = kJpegHuffmanMaxBitLength;
    while (max_length > 0 && counts[max_length] == 0) --max_length;
    --counts[max_length];
    int total_count = 0;
    for (int j = 0; j <= max_length; ++j) total_count += counts[j];
    data[pos++] = is_dc ? i : i - num_dc_histo + 0x10;
    for (size_t j = 1; j <= kJpegHuffmanMaxBitLength; ++j) {
      data[pos++] = counts[j];
    }
    for (size_t j = 0; j < total_count; ++j) {
      data[pos++] = values[j];
    }
  }

  // Emit SOS marker data.
  data[pos++] = 0xff;
  data[pos++] = 0xda;
  data[pos++] = sos_marker_len >> 8;
  data[pos++] = sos_marker_len & 0xff;
  data[pos++] = ncomps;
  for (int i = 0; i < ncomps; ++i) {
    data[pos++] = jpg.components[i].id;
    data[pos++] = (dc_histo_indexes[i] << 4) | ac_histo_indexes[i];
  }
  data[pos++] = 0;
  data[pos++] = 63;
  data[pos++] = 0;
  assert(pos == data.size());
  return JPEGWrite(out, &data[0], data.size());
}

void EncodeDCTBlockSequential(const coeff_t* coeffs,
                              const HuffmanCodeTable& dc_huff,
                              const HuffmanCodeTable& ac_huff,
                              coeff_t* last_dc_coeff,
                              BitWriter* bw) {
  coeff_t temp2;
  coeff_t temp;
  temp2 = coeffs[0];
  temp = temp2 - *last_dc_coeff;
  *last_dc_coeff = temp2;
  temp2 = temp;
  if (temp < 0) {
    temp = -temp;
    temp2--;
  }
  int nbits = Log2Floor(temp) + 1;
  bw->WriteBits(dc_huff.depth[nbits], dc_huff.code[nbits]);
  if (nbits > 0) {
    bw->WriteBits(nbits, temp2 & ((1 << nbits) - 1));
  }
  int r = 0;
  for (int k = 1; k < 64; ++k) {
    if ((temp = coeffs[kJPEGNaturalOrder[k]]) == 0) {
      r++;
      continue;
    }
    if (temp < 0) {
      temp = -temp;
      temp2 = ~temp;
    } else {
      temp2 = temp;
    }
    while (r > 15) {
      bw->WriteBits(ac_huff.depth[0xf0], ac_huff.code[0xf0]);
      r -= 16;
    }
    int nbits = Log2FloorNonZero(temp) + 1;
    int symbol = (r << 4) + nbits;
    bw->WriteBits(ac_huff.depth[symbol], ac_huff.code[symbol]);
    bw->WriteBits(nbits, temp2 & ((1 << nbits) - 1));
    r = 0;
  }
  if (r > 0) {
    bw->WriteBits(ac_huff.depth[0], ac_huff.code[0]);
  }
}

bool EncodeScan(const JPEGData& jpg,
                const std::vector<HuffmanCodeTable>& dc_huff_table,
                const std::vector<HuffmanCodeTable>& ac_huff_table,
                JPEGOutput out) {
  coeff_t last_dc_coeff[kMaxComponents] = { 0 };
  BitWriter bw(1 << 17);
  for (int mcu_y = 0; mcu_y < jpg.MCU_rows; ++mcu_y) {
    for (int mcu_x = 0; mcu_x < jpg.MCU_cols; ++mcu_x) {
      // Encode one MCU
      for (int i = 0; i < jpg.components.size(); ++i) {
        const JPEGComponent& c = jpg.components[i];
        int nblocks_y = c.v_samp_factor;
        int nblocks_x = c.h_samp_factor;
        for (int iy = 0; iy < nblocks_y; ++iy) {
          for (int ix = 0; ix < nblocks_x; ++ix) {
            int block_y = mcu_y * nblocks_y + iy;
            int block_x = mcu_x * nblocks_x + ix;
            int block_idx = block_y * c.width_in_blocks + block_x;
            const coeff_t* coeffs = &c.coeffs[block_idx << 6];
            EncodeDCTBlockSequential(coeffs, dc_huff_table[i], ac_huff_table[i],
                                     &last_dc_coeff[i], &bw);
          }
        }
      }
      if (bw.pos > (1 << 16)) {
        if (!JPEGWrite(out, bw.data.get(), bw.pos)) {
          return false;
        }
        bw.pos = 0;
      }
    }
  }
  bw.JumpToByteBoundary();
  return !bw.overflow && JPEGWrite(out, bw.data.get(), bw.pos);
}

}  // namespace

bool WriteJpeg(const JPEGData& jpg, bool strip_metadata, JPEGOutput out) {
  static const uint8_t kSOIMarker[2] = { 0xff, 0xd8 };
  static const uint8_t kEOIMarker[2] = { 0xff, 0xd9 };
  std::vector<HuffmanCodeTable> dc_codes;
  std::vector<HuffmanCodeTable> ac_codes;
  return (JPEGWrite(out, kSOIMarker, sizeof(kSOIMarker)) &&
          EncodeMetadata(jpg, strip_metadata, out) &&
          EncodeDQT(jpg.quant, out) &&
          EncodeSOF(jpg, out) &&
          BuildAndEncodeHuffmanCodes(jpg, out, &dc_codes, &ac_codes) &&
          EncodeScan(jpg, dc_codes, ac_codes, out) &&
          JPEGWrite(out, kEOIMarker, sizeof(kEOIMarker)) &&
          (strip_metadata || JPEGWrite(out, jpg.tail_data)));
}

int NullOut(void* data, const uint8_t* buf, size_t count) {
  return count;
}

void BuildSequentialHuffmanCodes(
    const JPEGData& jpg,
    std::vector<HuffmanCodeTable>* dc_huffman_code_tables,
    std::vector<HuffmanCodeTable>* ac_huffman_code_tables) {
  JPEGOutput out(NullOut, nullptr);
  BuildAndEncodeHuffmanCodes(jpg, out, dc_huffman_code_tables,
                             ac_huffman_code_tables);
}

}  // namespace guetzli
