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

// Entropy encoding (Huffman) utilities.

#ifndef GUETZLI_ENTROPY_ENCODE_H_
#define GUETZLI_ENTROPY_ENCODE_H_

#include <stddef.h>
#include <stdint.h>

namespace guetzli {

// A node of a Huffman tree.
struct HuffmanTree {
  HuffmanTree() {}
  HuffmanTree(uint32_t count, int16_t left, int16_t right)
      : total_count_(count),
        index_left_(left),
        index_right_or_value_(right) {
  }
  uint32_t total_count_;
  int16_t index_left_;
  int16_t index_right_or_value_;
};

bool SetDepth(int p, HuffmanTree *pool, uint8_t *depth, int max_depth);

// This function will create a Huffman tree.
//
// The (data,length) contains the population counts.
// The tree_limit is the maximum bit depth of the Huffman codes.
//
// The depth contains the tree, i.e., how many bits are used for
// the symbol.
//
// The actual Huffman tree is constructed in the tree[] array, which has to
// be at least 2 * length + 1 long.
//
// See http://en.wikipedia.org/wiki/Huffman_coding
void CreateHuffmanTree(const uint32_t *data,
                       const size_t length,
                       const int tree_limit,
                       HuffmanTree* tree,
                       uint8_t *depth);

}  // namespace guetzli

#endif  // GUETZLI_ENTROPY_ENCODE_H_
