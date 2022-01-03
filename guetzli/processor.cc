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

#include "guetzli/processor.h"

#include <algorithm>
#include <set>
#include <string.h>
#include <vector>

#include "guetzli/butteraugli_comparator.h"
#include "guetzli/comparator.h"
#include "guetzli/debug_print.h"
#include "guetzli/fast_log.h"
#include "guetzli/jpeg_data_decoder.h"
#include "guetzli/jpeg_data_encoder.h"
#include "guetzli/jpeg_data_reader.h"
#include "guetzli/jpeg_data_writer.h"
#include "guetzli/output_image.h"
#include "guetzli/quantize.h"

namespace guetzli {

namespace {

static const size_t kBlockSize = 3 * kDCTBlockSize;

struct CoeffData {
  int idx;
  float block_err;
};
struct QuantData {
  int q[3][kDCTBlockSize];
  size_t jpg_size;
  bool dist_ok;
};
class Processor {
 public:
  bool ProcessJpegData(const Params& params, const JPEGData& jpg_in,
                       Comparator* comparator, GuetzliOutput* out,
                       ProcessStats* stats);

 private:
  void SelectFrequencyMasking(const JPEGData& jpg, OutputImage* img,
                              const uint8_t comp_mask, const double target_mul,
                              bool stop_early);
  void ComputeBlockZeroingOrder(
      const coeff_t block[kBlockSize], const coeff_t orig_block[kBlockSize],
      const int block_x, const int block_y, const int factor_x,
      const int factor_y, const uint8_t comp_mask, OutputImage* img,
      std::vector<CoeffData>* output_order);
  bool SelectQuantMatrix(const JPEGData& jpg_in, const bool downsample,
                         int best_q[3][kDCTBlockSize],
                         OutputImage* img);
  QuantData TryQuantMatrix(const JPEGData& jpg_in,
                           const float target_mul,
                           int q[3][kDCTBlockSize],
                           OutputImage* img);
  void MaybeOutput(const std::string& encoded_jpg);
  void DownsampleImage(OutputImage* img);
  void OutputJpeg(const JPEGData& in, std::string* out);

  Params params_;
  Comparator* comparator_;
  GuetzliOutput* final_output_;
  ProcessStats* stats_;
};

void RemoveOriginalQuantization(JPEGData* jpg, int q_in[3][kDCTBlockSize]) {
  for (int i = 0; i < 3; ++i) {
    JPEGComponent& c = jpg->components[i];
    const int* q = &jpg->quant[c.quant_idx].values[0];
    memcpy(&q_in[i][0], q, kDCTBlockSize * sizeof(q[0]));
    for (size_t j = 0; j < c.coeffs.size(); ++j) {
      c.coeffs[j] *= q[j % kDCTBlockSize];
    }
  }
  int q[3][kDCTBlockSize];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < kDCTBlockSize; ++j) q[i][j] = 1;
  SaveQuantTables(q, jpg);
}

void Processor::DownsampleImage(OutputImage* img) {
  if (img->component(1).factor_x() > 1 || img->component(1).factor_y() > 1) {
    return;
  }
  OutputImage::DownsampleConfig cfg;
  cfg.use_silver_screen = params_.use_silver_screen;
  img->Downsample(cfg);
}

bool CheckJpegSanity(const JPEGData& jpg) {
  const int kMaxComponent = 1 << 12;
  for (const JPEGComponent& comp : jpg.components) {
    const JPEGQuantTable& quant_table = jpg.quant[comp.quant_idx];
    for (int i = 0; i < comp.coeffs.size(); i++) {
      coeff_t coeff = comp.coeffs[i];
      int quant = quant_table.values[i % kDCTBlockSize];
      if (std::abs(static_cast<int64_t>(coeff) * quant) > kMaxComponent) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace

int GuetzliStringOut(void* data, const uint8_t* buf, size_t count) {
  std::string* sink =
      reinterpret_cast<std::string*>(data);
  sink->append(reinterpret_cast<const char*>(buf), count);
  return count;
}

void Processor::OutputJpeg(const JPEGData& jpg,
                           std::string* out) {
  out->clear();
  JPEGOutput output(GuetzliStringOut, out);
  if (!WriteJpeg(jpg, params_.clear_metadata, output)) {
    assert(0);
  }
}

void Processor::MaybeOutput(const std::string& encoded_jpg) {
  double score = comparator_->ScoreOutputSize(encoded_jpg.size());
  GUETZLI_LOG(stats_, " Score[%.4f]", score);
  if (score < final_output_->score || final_output_->score < 0) {
    final_output_->jpeg_data = encoded_jpg;
    final_output_->score = score;
    GUETZLI_LOG(stats_, " (*)");
  }
  GUETZLI_LOG(stats_, "\n");
}

bool CompareQuantData(const QuantData& a, const QuantData& b) {
  if (a.dist_ok && !b.dist_ok) return true;
  if (!a.dist_ok && b.dist_ok) return false;
  return a.jpg_size < b.jpg_size;
}

// Compares a[0..kBlockSize) and b[0..kBlockSize) vectors, and returns
//   0 : if they are equal
//  -1 : if a is everywhere <= than b and in at least one coordinate <
//   1 : if a is everywhere >= than b and in at least one coordinate >
//   2 : if a and b are uncomparable (some coordinate smaller and some greater)
int CompareQuantMatrices(const int* a, const int* b) {
  int i = 0;
  while (i < kBlockSize && a[i] == b[i]) ++i;
  if (i == kBlockSize) {
    return 0;
  }
  if (a[i] < b[i]) {
    for (++i; i < kBlockSize; ++i) {
      if (a[i] > b[i]) return 2;
    }
    return -1;
  } else {
    for (++i; i < kBlockSize; ++i) {
      if (a[i] < b[i]) return 2;
    }
    return 1;
  }
}

double ContrastSensitivity(int k) {
  return 1.0 / (1.0 + kJPEGZigZagOrder[k] / 2.0);
}

double QuantMatrixHeuristicScore(const int q[3][kDCTBlockSize]) {
  double score = 0.0;
  for (int c = 0; c < 3; ++c) {
    for (int k = 0; k < kDCTBlockSize; ++k) {
      score += 0.5 * (q[c][k] - 1.0) * ContrastSensitivity(k);
    }
  }
  return score;
}

class QuantMatrixGenerator {
 public:
  QuantMatrixGenerator(bool downsample, ProcessStats* stats)
      : downsample_(downsample), hscore_a_(-1.0), hscore_b_(-1.0),
        total_csf_(0.0), stats_(stats) {
    for (int k = 0; k < kDCTBlockSize; ++k) {
      total_csf_ += 3.0 * ContrastSensitivity(k);
    }
  }

  bool GetNext(int q[3][kDCTBlockSize]) {
    // This loop should terminate by return. This 1000 iteration limit is just a
    // precaution.
    for (int iter = 0; iter < 1000; iter++) {
      double hscore;
      if (hscore_b_ == -1.0) {
        if (hscore_a_ == -1.0) {
          hscore = downsample_ ? 0.0 : total_csf_;
        } else {
          if (hscore_a_ < 5.0 * total_csf_) {
            hscore = hscore_a_ + total_csf_;
          } else {
            hscore = 2 * (hscore_a_ + total_csf_);
          }
        }
        if (hscore > 100 * total_csf_) {
          // We could not find a quantization matrix that creates enough
          // butteraugli error. This can happen if all dct coefficients are
          // close to zero in the original image.
          return false;
        }
      } else if (hscore_b_ == 0.0) {
        return false;
      } else if (hscore_a_ == -1.0) {
        hscore = 0.0;
      } else {
        int lower_q[3][kDCTBlockSize];
        int upper_q[3][kDCTBlockSize];
        constexpr double kEps = 0.05;
        GetQuantMatrixWithHeuristicScore(
            (1 - kEps) * hscore_a_ + kEps * 0.5 * (hscore_a_ + hscore_b_),
            lower_q);
        GetQuantMatrixWithHeuristicScore(
            (1 - kEps) * hscore_b_ + kEps * 0.5 * (hscore_a_ + hscore_b_),
            upper_q);
        if (CompareQuantMatrices(&lower_q[0][0], &upper_q[0][0]) == 0)
          return false;
        hscore = (hscore_a_ + hscore_b_) * 0.5;
      }
      GetQuantMatrixWithHeuristicScore(hscore, q);
      bool retry = false;
      for (size_t i = 0; i < quants_.size(); ++i) {
        if (CompareQuantMatrices(&q[0][0], &quants_[i].q[0][0]) == 0) {
          if (quants_[i].dist_ok) {
            hscore_a_ = hscore;
          } else {
            hscore_b_ = hscore;
          }
          retry = true;
          break;
        }
      }
      if (!retry) return true;
    }
    return false;
  }

  void Add(const QuantData& data) {
    quants_.push_back(data);
    double hscore = QuantMatrixHeuristicScore(data.q);
    if (data.dist_ok) {
      hscore_a_ = std::max(hscore_a_, hscore);
    } else {
      hscore_b_ = hscore_b_ == -1.0 ? hscore : std::min(hscore_b_, hscore);
    }
  }

 private:
  void GetQuantMatrixWithHeuristicScore(double score,
                                        int q[3][kDCTBlockSize]) const {
    int level = static_cast<int>(score / total_csf_);
    score -= level * total_csf_;
    for (int k = kDCTBlockSize - 1; k >= 0; --k) {
      for (int c = 0; c < 3; ++c) {
        q[c][kJPEGNaturalOrder[k]] = 2 * level + (score > 0.0 ? 3 : 1);
      }
      score -= 3.0 * ContrastSensitivity(kJPEGNaturalOrder[k]);
    }
  }

  const bool downsample_;
  // Lower bound for quant matrix heuristic score used in binary search.
  double hscore_a_;
  // Upper bound for quant matrix heuristic score used in binary search, or 0.0
  // if no upper bound is found yet.
  double hscore_b_;
  // Cached value of the sum of all ContrastSensitivity() values over all
  // quant matrix elements.
  double total_csf_;
  std::vector<QuantData> quants_;

  ProcessStats* stats_;
};

QuantData Processor::TryQuantMatrix(const JPEGData& jpg_in,
                                    const float target_mul,
                                    int q[3][kDCTBlockSize],
                                    OutputImage* img) {
  QuantData data;
  memcpy(data.q, q, sizeof(data.q));
  img->CopyFromJpegData(jpg_in);
  img->ApplyGlobalQuantization(data.q);
  std::string encoded_jpg;
  {
    JPEGData jpg_out = jpg_in;
    img->SaveToJpegData(&jpg_out);
    OutputJpeg(jpg_out, &encoded_jpg);
  }
  GUETZLI_LOG(stats_, "Iter %2d: %s quantization matrix:\n",
              stats_->counters[kNumItersCnt] + 1,
              img->FrameTypeStr().c_str());
  GUETZLI_LOG_QUANT(stats_, q);
  GUETZLI_LOG(stats_, "Iter %2d: %s GQ[%5.2f] Out[%7zd]",
              stats_->counters[kNumItersCnt] + 1,
              img->FrameTypeStr().c_str(),
              QuantMatrixHeuristicScore(q), encoded_jpg.size());
  ++stats_->counters[kNumItersCnt];
  comparator_->Compare(*img);
  data.dist_ok = comparator_->DistanceOK(target_mul);
  data.jpg_size = encoded_jpg.size();
  MaybeOutput(encoded_jpg);
  return data;
}

bool Processor::SelectQuantMatrix(const JPEGData& jpg_in, const bool downsample,
                                  int best_q[3][kDCTBlockSize],
                                  OutputImage* img) {
  QuantMatrixGenerator qgen(downsample, stats_);
  // Don't try to go up to exactly the target distance when selecting a
  // quantization matrix, since we will need some slack to do the frequency
  // masking later.
  const float target_mul_high = 0.97f;
  const float target_mul_low = 0.95f;

  QuantData best = TryQuantMatrix(jpg_in, target_mul_high, best_q, img);
  for (;;) {
    int q_next[3][kDCTBlockSize];
    if (!qgen.GetNext(q_next)) {
      break;
    }

    QuantData data = TryQuantMatrix(jpg_in, target_mul_high, q_next, img);
    qgen.Add(data);
    if (CompareQuantData(data, best)) {
      best = data;
      if (data.dist_ok && !comparator_->DistanceOK(target_mul_low)) {
        break;
      }
    }
  }

  memcpy(&best_q[0][0], &best.q[0][0], kBlockSize * sizeof(best_q[0][0]));
  GUETZLI_LOG(stats_, "\n%s selected quantization matrix:\n",
              downsample ? "YUV420" : "YUV444");
  GUETZLI_LOG_QUANT(stats_, best_q);
  return best.dist_ok;
}


// REQUIRES: block[c*64...(c*64+63)] is all zero if (comp_mask & (1<<c)) == 0.
void Processor::ComputeBlockZeroingOrder(
    const coeff_t block[kBlockSize], const coeff_t orig_block[kBlockSize],
    const int block_x, const int block_y, const int factor_x,
    const int factor_y, const uint8_t comp_mask, OutputImage* img,
    std::vector<CoeffData>* output_order) {
  static const uint8_t oldCsf[kDCTBlockSize] = {
      10, 10, 20, 40, 60, 70, 80, 90,
      10, 20, 30, 60, 70, 80, 90, 90,
      20, 30, 60, 70, 80, 90, 90, 90,
      40, 60, 70, 80, 90, 90, 90, 90,
      60, 70, 80, 90, 90, 90, 90, 90,
      70, 80, 90, 90, 90, 90, 90, 90,
      80, 90, 90, 90, 90, 90, 90, 90,
      90, 90, 90, 90, 90, 90, 90, 90,
  };
  static const double kWeight[3] = { 1.0, 0.22, 0.20 };
#include "guetzli/order.inc"
  std::vector<std::pair<int, float> > input_order;
  for (int c = 0; c < 3; ++c) {
    if (!(comp_mask & (1 << c))) continue;
    for (int k = 1; k < kDCTBlockSize; ++k) {
      int idx = c * kDCTBlockSize + k;
      if (block[idx] != 0) {
        float score;
        if (params_.new_zeroing_model) {
          score = std::abs(orig_block[idx]) * csf[idx] + bias[idx];
        } else {
          score = static_cast<float>((std::abs(orig_block[idx]) - kJPEGZigZagOrder[k] / 64.0) *
                  kWeight[c] / oldCsf[k]);
        }
        input_order.push_back(std::make_pair(idx, score));
      }
    }
  }
  std::sort(input_order.begin(), input_order.end(),
            [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
              return a.second < b.second; });
  coeff_t processed_block[kBlockSize];
  memcpy(processed_block, block, sizeof(processed_block));
  comparator_->SwitchBlock(block_x, block_y, factor_x, factor_y);
  while (!input_order.empty()) {
    float best_err = 1e17f;
    int best_i = 0;
    for (size_t i = 0; i < std::min<size_t>(params_.zeroing_greedy_lookahead,
                                         input_order.size());
         ++i) {
      coeff_t candidate_block[kBlockSize];
      memcpy(candidate_block, processed_block, sizeof(candidate_block));
      const int idx = input_order[i].first;
      candidate_block[idx] = 0;
      for (int c = 0; c < 3; ++c) {
        if (comp_mask & (1 << c)) {
          img->component(c).SetCoeffBlock(
              block_x, block_y, &candidate_block[c * kDCTBlockSize]);
        }
      }
      float max_err = 0;
      for (int iy = 0; iy < factor_y; ++iy) {
        for (int ix = 0; ix < factor_x; ++ix) {
          int block_xx = block_x * factor_x + ix;
          int block_yy = block_y * factor_y + iy;
          if (8 * block_xx < img->width() && 8 * block_yy < img->height()) {
            float err = static_cast<float>(comparator_->CompareBlock(*img, ix, iy));
            max_err = std::max(max_err, err);
          }
        }
      }
      if (max_err < best_err) {
        best_err = max_err;
        best_i = i;
      }
    }
    int idx = input_order[best_i].first;
    processed_block[idx] = 0;
    input_order.erase(input_order.begin() + best_i);
    output_order->push_back({idx, best_err});
    for (int c = 0; c < 3; ++c) {
      if (comp_mask & (1 << c)) {
        img->component(c).SetCoeffBlock(
            block_x, block_y, &processed_block[c * kDCTBlockSize]);
      }
    }
  }
  // Make the block error values monotonic.
  float min_err = 1e10;
  for (int i = output_order->size() - 1; i >= 0; --i) {
    min_err = std::min(min_err, (*output_order)[i].block_err);
    (*output_order)[i].block_err = min_err;
  }
  // Cut off at the block error limit.
  size_t num = 0;
  while (num < output_order->size() &&
         (*output_order)[num].block_err <= comparator_->BlockErrorLimit()) {
    ++num;
  }
  output_order->resize(num);
  // Restore *img to the same state as it was at the start of this function.
  for (int c = 0; c < 3; ++c) {
    if (comp_mask & (1 << c)) {
      img->component(c).SetCoeffBlock(
          block_x, block_y, &block[c * kDCTBlockSize]);
    }
  }
}

namespace {

void UpdateACHistogram(const int weight,
                       const coeff_t* coeffs,
                       const int* q,
                       JpegHistogram* ac_histogram) {
  int r = 0;
  for (int k = 1; k < 64; ++k) {
    const int k_nat = kJPEGNaturalOrder[k];
    coeff_t coeff = coeffs[k_nat];
    if (coeff == 0) {
      r++;
      continue;
    }
    while (r > 15) {
      ac_histogram->Add(0xf0, weight);
      r -= 16;
    }
    int nbits = Log2FloorNonZero(std::abs(coeff / q[k_nat])) + 1;
    int symbol = (r << 4) + nbits;
    ac_histogram->Add(symbol, weight);
    r = 0;
  }
  if (r > 0) {
    ac_histogram->Add(0, weight);
  }
}

size_t ComputeEntropyCodes(const std::vector<JpegHistogram>& histograms,
                           std::vector<uint8_t>* depths) {
  std::vector<JpegHistogram> clustered = histograms;
  size_t num = histograms.size();
  std::vector<int> indexes(histograms.size());
  std::vector<uint8_t> clustered_depths(
      histograms.size() * JpegHistogram::kSize);
  ClusterHistograms(&clustered[0], &num, &indexes[0], &clustered_depths[0]);
  depths->resize(clustered_depths.size());
  for (size_t i = 0; i < histograms.size(); ++i) {
    memcpy(&(*depths)[i * JpegHistogram::kSize],
           &clustered_depths[indexes[i] * JpegHistogram::kSize],
           JpegHistogram::kSize);
  }
  size_t histogram_size = 0;
  for (size_t i = 0; i < num; ++i) {
    histogram_size += HistogramHeaderCost(clustered[i]) / 8;
  }
  return histogram_size;
}

size_t EntropyCodedDataSize(const std::vector<JpegHistogram>& histograms,
                            const std::vector<uint8_t>& depths) {
  size_t numbits = 0;
  for (size_t i = 0; i < histograms.size(); ++i) {
    numbits += HistogramEntropyCost(
        histograms[i], &depths[i * JpegHistogram::kSize]);
  }
  return (numbits + 7) / 8;
}

size_t EstimateDCSize(const JPEGData& jpg) {
  std::vector<JpegHistogram> histograms(jpg.components.size());
  BuildDCHistograms(jpg, &histograms[0]);
  size_t num = histograms.size();
  std::vector<int> indexes(num);
  std::vector<uint8_t> depths(num * JpegHistogram::kSize);
  return ClusterHistograms(&histograms[0], &num, &indexes[0], &depths[0]);
}

}  // namespace

void Processor::SelectFrequencyMasking(const JPEGData& jpg, OutputImage* img,
                                       const uint8_t comp_mask,
                                       const double target_mul,
                                       bool stop_early) {
  const int width = img->width();
  const int height = img->height();
  const int ncomp = jpg.components.size();
  const int last_c = Log2FloorNonZero(comp_mask);
  if (static_cast<size_t>(last_c) >= jpg.components.size()) return;
  const int factor_x = img->component(last_c).factor_x();
  const int factor_y = img->component(last_c).factor_y();
  const int block_width = (width + 8 * factor_x - 1) / (8 * factor_x);
  const int block_height = (height + 8 * factor_y - 1) / (8 * factor_y);
  const int num_blocks = block_width * block_height;

  std::vector<int> candidate_coeff_offsets(num_blocks + 1);
  std::vector<uint8_t> candidate_coeffs;
  std::vector<float> candidate_coeff_errors;
  candidate_coeffs.reserve(60 * num_blocks);
  candidate_coeff_errors.reserve(60 * num_blocks);
  std::vector<CoeffData> block_order;
  block_order.reserve(3 * kDCTBlockSize);
  comparator_->StartBlockComparisons();
  for (int block_y = 0, block_ix = 0; block_y < block_height; ++block_y) {
    for (int block_x = 0; block_x < block_width; ++block_x, ++block_ix) {
      coeff_t block[kBlockSize] = { 0 };
      coeff_t orig_block[kBlockSize] = { 0 };
      for (int c = 0; c < 3; ++c) {
        if (comp_mask & (1 << c)) {
          assert(img->component(c).factor_x() == factor_x);
          assert(img->component(c).factor_y() == factor_y);
          img->component(c).GetCoeffBlock(block_x, block_y,
                                          &block[c * kDCTBlockSize]);
          const JPEGComponent& comp = jpg.components[c];
          int jpg_block_ix = block_y * comp.width_in_blocks + block_x;
          memcpy(&orig_block[c * kDCTBlockSize],
                 &comp.coeffs[jpg_block_ix * kDCTBlockSize],
                 kDCTBlockSize * sizeof(orig_block[0]));
        }
      }
      block_order.clear();
      ComputeBlockZeroingOrder(block, orig_block, block_x, block_y, factor_x,
                               factor_y, comp_mask, img, &block_order);
      candidate_coeff_offsets[block_ix] = candidate_coeffs.size();
      for (size_t i = 0; i < block_order.size(); ++i) {
        candidate_coeffs.push_back(block_order[i].idx);
        candidate_coeff_errors.push_back(block_order[i].block_err);
      }
    }
  }
  comparator_->FinishBlockComparisons();
  candidate_coeff_offsets[num_blocks] = candidate_coeffs.size();

  std::vector<JpegHistogram> ac_histograms(ncomp);
  int jpg_header_size, dc_size;
  {
    JPEGData jpg_out = jpg;
    img->SaveToJpegData(&jpg_out);
    jpg_header_size = JpegHeaderSize(jpg_out, params_.clear_metadata);
    dc_size = EstimateDCSize(jpg_out);
    BuildACHistograms(jpg_out, &ac_histograms[0]);
  }
  std::vector<uint8_t> ac_depths;
  int ac_histogram_size = ComputeEntropyCodes(ac_histograms, &ac_depths);
  int base_size = jpg_header_size + dc_size + ac_histogram_size +
      EntropyCodedDataSize(ac_histograms, ac_depths);
  int prev_size = base_size;

  std::vector<float> max_block_error(num_blocks);
  std::vector<int> last_indexes(num_blocks);

  bool first_up_iter = true;
  for (int direction : {1, -1}) {
    for (;;) {
      if (stop_early && direction == -1) {
        if (prev_size > 1.01 * final_output_->jpeg_data.size()) {
          // If we are down-adjusting the error, the output size will only keep
          // increasing.
          // TODO(user): Do this check always by comparing only the size
          // of the currently processed components.
          break;
        }
      }
      std::vector<std::pair<int, float> > global_order;
      int blocks_to_change;
      std::vector<float> block_weight;
      for (int rblock = 1; rblock <= 4; ++rblock) {
        block_weight = std::vector<float>(num_blocks);
        std::vector<float> distmap(width * height);
        if (!first_up_iter) {
          distmap = comparator_->distmap();
        }
        comparator_->ComputeBlockErrorAdjustmentWeights(
            direction, rblock, target_mul, factor_x, factor_y, distmap,
            &block_weight);
        global_order.clear();
        blocks_to_change = 0;
        for (int block_y = 0, block_ix = 0; block_y < block_height; ++block_y) {
          for (int block_x = 0; block_x < block_width; ++block_x, ++block_ix) {
            const int last_index = last_indexes[block_ix];
            const int offset = candidate_coeff_offsets[block_ix];
            const int num_candidates =
                candidate_coeff_offsets[block_ix + 1] - offset;
            const float* candidate_errors = &candidate_coeff_errors[offset];
            const float max_err = max_block_error[block_ix];
            if (block_weight[block_ix] == 0) {
              continue;
            }
            if (direction > 0) {
              for (size_t i = last_index; i < num_candidates; ++i) {
                float val = ((candidate_errors[i] - max_err) /
                             block_weight[block_ix]);
                global_order.push_back(std::make_pair(block_ix, val));
              }
              blocks_to_change += (last_index < num_candidates ? 1 : 0);
            } else {
              for (int i = last_index - 1; i >= 0; --i) {
                float val = ((max_err - candidate_errors[i]) /
                             block_weight[block_ix]);
                global_order.push_back(std::make_pair(block_ix, val));
              }
              blocks_to_change += (last_index > 0 ? 1 : 0);
            }
          }
        }
        if (!global_order.empty()) {
          // If we found something to adjust with the current block adjustment
          // radius, we can stop and adjust the blocks we have.
          break;
        }
      }

      if (global_order.empty()) {
        break;
      }

      std::sort(global_order.begin(), global_order.end(),
                [](const std::pair<int, float>& a,
                   const std::pair<int, float>& b) {
                  return a.second < b.second; });

      double rel_size_delta = direction > 0 ? 0.01 : 0.0005;
      if (direction > 0 && comparator_->DistanceOK(1.0)) {
        rel_size_delta = 0.05;
      }
      double min_size_delta = base_size * rel_size_delta;

      float coeffs_to_change_per_block =
          direction > 0 ? 2.0f : factor_x * factor_y * 0.2f;
      int min_coeffs_to_change = coeffs_to_change_per_block * blocks_to_change;

      if (first_up_iter) {
        const float limit = 0.75f * comparator_->BlockErrorLimit();
        auto it = std::partition_point(global_order.begin(), global_order.end(),
                                       [=](const std::pair<int, float>& a) {
                                         return a.second < limit; });
        min_coeffs_to_change = std::max<int>(min_coeffs_to_change,
                                             it - global_order.begin());
        first_up_iter = false;
      }

      std::set<int> changed_blocks;
      float val_threshold = 0.0;
      int changed_coeffs = 0;
      int est_jpg_size = prev_size;
      for (size_t i = 0; i < global_order.size(); ++i) {
        const int block_ix = global_order[i].first;
        const int block_x = block_ix % block_width;
        const int block_y = block_ix / block_width;
        const int last_idx = last_indexes[block_ix];
        const int offset = candidate_coeff_offsets[block_ix];
        const uint8_t* candidates = &candidate_coeffs[offset];
        const int idx = candidates[last_idx + std::min(direction, 0)];
        const int c = idx / kDCTBlockSize;
        const int k = idx % kDCTBlockSize;
        const int* quant = img->component(c).quant();
        const JPEGComponent& comp = jpg.components[c];
        const int jpg_block_ix = block_y * comp.width_in_blocks + block_x;
        const int newval = direction > 0 ? 0 : Quantize(
            comp.coeffs[jpg_block_ix * kDCTBlockSize + k], quant[k]);
        coeff_t block[kDCTBlockSize] = { 0 };
        img->component(c).GetCoeffBlock(block_x, block_y, block);
        UpdateACHistogram(-1, block, quant, &ac_histograms[c]);
        double sum_of_hf = 0;
        for (int ii = 3; ii < 64; ++ii) {
          if ((ii & 7) < 3 && ii < 3 * 8) continue;
          sum_of_hf += std::abs(comp.coeffs[jpg_block_ix * kDCTBlockSize + ii]);
        }
        int limit = sum_of_hf < 60 ? 4 : 8;
        bool precious =
            (k == 1 || k == 8) &&
            std::abs(comp.coeffs[jpg_block_ix * kDCTBlockSize + k]) >= limit;
        if (!precious || newval != 0) {
          block[k] = newval;
        }
        UpdateACHistogram(1, block, quant, &ac_histograms[c]);
        img->component(c).SetCoeffBlock(block_x, block_y, block);
        last_indexes[block_ix] += direction;
        changed_blocks.insert(block_ix);
        val_threshold = global_order[i].second;
        ++changed_coeffs;
        static const int kEntropyCodeUpdateFreq = 10;
        if (i % kEntropyCodeUpdateFreq == 0) {
          ac_histogram_size = ComputeEntropyCodes(ac_histograms, &ac_depths);
        }
        est_jpg_size = jpg_header_size + dc_size + ac_histogram_size +
            EntropyCodedDataSize(ac_histograms, ac_depths);
        if (changed_coeffs > min_coeffs_to_change &&
            std::abs(est_jpg_size - prev_size) > min_size_delta) {
          break;
        }
      }
      size_t global_order_size = global_order.size();
      std::vector<std::pair<int, float>>().swap(global_order);

      for (int i = 0; i < num_blocks; ++i) {
        max_block_error[i] += block_weight[i] * val_threshold * direction;
      }

      ++stats_->counters[kNumItersCnt];
      ++stats_->counters[direction > 0 ? kNumItersUpCnt : kNumItersDownCnt];
      std::string encoded_jpg;
      {
        JPEGData jpg_out = jpg;
        img->SaveToJpegData(&jpg_out);
        OutputJpeg(jpg_out, &encoded_jpg);
      }
      GUETZLI_LOG(stats_,
                  "Iter %2d: %s(%d) %s Coeffs[%d/%zd] "
                  "Blocks[%zd/%d/%d] ValThres[%.4f] Out[%7zd] EstErr[%.2f%%]",
                  stats_->counters[kNumItersCnt], img->FrameTypeStr().c_str(),
                  comp_mask, direction > 0 ? "up" : "down", changed_coeffs,
                  global_order_size, changed_blocks.size(),
                  blocks_to_change, num_blocks, val_threshold,
                  encoded_jpg.size(),
                  100.0 - (100.0 * est_jpg_size) / encoded_jpg.size());
      comparator_->Compare(*img);
      MaybeOutput(encoded_jpg);
      prev_size = est_jpg_size;
    }
  }
}

bool IsGrayscale(const JPEGData& jpg) {
  for (int c = 1; c < 3; ++c) {
    const JPEGComponent& comp = jpg.components[c];
    for (size_t i = 0; i < comp.coeffs.size(); ++i) {
      if (comp.coeffs[i] != 0) return false;
    }
  }
  return true;
}

bool Processor::ProcessJpegData(const Params& params, const JPEGData& jpg_in,
                                Comparator* comparator, GuetzliOutput* out,
                                ProcessStats* stats) {
  params_ = params;
  comparator_ = comparator;
  final_output_ = out;
  stats_ = stats;

  if (params.butteraugli_target > 2.0f) {
    fprintf(stderr,
            "Guetzli should be called with quality >= 84, otherwise the\n"
            "output will have noticeable artifacts. If you want to\n"
            "proceed anyway, please edit the source code.\n");
    return false;
  }
  if (jpg_in.components.size() != 3 || !HasYCbCrColorSpace(jpg_in)) {
    fprintf(stderr, "Only YUV color space input jpeg is supported\n");
    return false;
  }
  bool input_is_420;
  if (jpg_in.Is444()) {
    input_is_420 = false;
  } else if (jpg_in.Is420()) {
    input_is_420 = true;
  } else {
    fprintf(stderr, "Unsupported sampling factors:");
    for (size_t i = 0; i < jpg_in.components.size(); ++i) {
      fprintf(stderr, " %dx%d", jpg_in.components[i].h_samp_factor,
              jpg_in.components[i].v_samp_factor);
    }
    fprintf(stderr, "\n");
    return false;
  }
  int q_in[3][kDCTBlockSize];
  // Output the original image, in case we do not manage to create anything
  // with a good enough quality.
  std::string encoded_jpg;
  OutputJpeg(jpg_in, &encoded_jpg);
  final_output_->score = -1;
  GUETZLI_LOG(stats, "Original Out[%7zd]", encoded_jpg.size());
  if (comparator_ == nullptr) {
    GUETZLI_LOG(stats, " <image too small for Butteraugli>\n");
    final_output_->jpeg_data = encoded_jpg;
    final_output_->score = encoded_jpg.size();
    // Butteraugli doesn't work with images this small.
    return true;
  }
  {
    JPEGData jpg = jpg_in;
    RemoveOriginalQuantization(&jpg, q_in);
    OutputImage img(jpg.width, jpg.height);
    img.CopyFromJpegData(jpg);
    comparator_->Compare(img);
  }
  MaybeOutput(encoded_jpg);
  int try_420 = (input_is_420 || params_.force_420 ||
                 (params_.try_420 && !IsGrayscale(jpg_in))) ? 1 : 0;
  int force_420 = (input_is_420 || params_.force_420) ? 1 : 0;
  for (int downsample = force_420; downsample <= try_420; ++downsample) {
    JPEGData jpg = jpg_in;
    RemoveOriginalQuantization(&jpg, q_in);
    OutputImage img(jpg.width, jpg.height);
    img.CopyFromJpegData(jpg);
    if (downsample) {
      DownsampleImage(&img);
      img.SaveToJpegData(&jpg);
    }
    int best_q[3][kDCTBlockSize];
    memcpy(best_q, q_in, sizeof(best_q));
    if (!SelectQuantMatrix(jpg, downsample != 0, best_q, &img)) {
      for (int c = 0; c < 3; ++c) {
        for (int i = 0; i < kDCTBlockSize; ++i) {
          best_q[c][i] = 1;
        }
      }
    }
    img.CopyFromJpegData(jpg);
    img.ApplyGlobalQuantization(best_q);

    if (!downsample) {
      SelectFrequencyMasking(jpg, &img, 7, 1.0, false);
    } else {
      const float ymul = jpg.components.size() == 1 ? 1.0f : 0.97f;
      SelectFrequencyMasking(jpg, &img, 1, ymul, false);
      SelectFrequencyMasking(jpg, &img, 6, 1.0, true);
    }
  }

  return true;
}

bool ProcessJpegData(const Params& params, const JPEGData& jpg_in,
                     Comparator* comparator, GuetzliOutput* out,
                     ProcessStats* stats) {
  Processor processor;
  return processor.ProcessJpegData(params, jpg_in, comparator, out, stats);
}

bool Process(const Params& params, ProcessStats* stats,
             const std::string& data,
             std::string* jpg_out) {
  JPEGData jpg;
  if (!ReadJpeg(data, JPEG_READ_ALL, &jpg)) {
    fprintf(stderr, "Can't read jpg data from input file\n");
    return false;
  }
  if (!CheckJpegSanity(jpg)) {
    fprintf(stderr, "Unsupported input JPEG (unexpectedly large coefficient "
            "values).\n");
    return false;
  }
  std::vector<uint8_t> rgb = DecodeJpegToRGB(jpg);
  if (rgb.empty()) {
    fprintf(stderr, "Unsupported input JPEG file (e.g. unsupported "
            "downsampling mode).\nPlease provide the input image as "
            "a PNG file.\n");
    return false;
  }
  GuetzliOutput out;
  ProcessStats dummy_stats;
  if (stats == nullptr) {
    stats = &dummy_stats;
  }
  std::unique_ptr<ButteraugliComparator> comparator;
  if (jpg.width >= 32 && jpg.height >= 32) {
    comparator.reset(
        new ButteraugliComparator(jpg.width, jpg.height, &rgb,
                                  params.butteraugli_target, stats));
  }
  bool ok = ProcessJpegData(params, jpg, comparator.get(), &out, stats);
  *jpg_out = out.jpeg_data;
  return ok;
}

bool Process(const Params& params, ProcessStats* stats,
             const std::vector<uint8_t>& rgb, int w, int h,
             std::string* jpg_out) {
  JPEGData jpg;
  if (!EncodeRGBToJpeg(rgb, w, h, &jpg)) {
    fprintf(stderr, "Could not create jpg data from rgb pixels\n");
    return false;
  }
  GuetzliOutput out;
  ProcessStats dummy_stats;
  if (stats == nullptr) {
    stats = &dummy_stats;
  }
  std::unique_ptr<ButteraugliComparator> comparator;
  if (jpg.width >= 32 && jpg.height >= 32) {
    comparator.reset(
        new ButteraugliComparator(jpg.width, jpg.height, &rgb,
                                  params.butteraugli_target, stats));
  }
  bool ok = ProcessJpegData(params, jpg, comparator.get(), &out, stats);
  *jpg_out = out.jpeg_data;
  return ok;
}

}  // namespace guetzli
