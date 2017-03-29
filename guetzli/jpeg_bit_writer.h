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

#ifndef GUETZLI_JPEG_BIT_WRITER_H_
#define GUETZLI_JPEG_BIT_WRITER_H_

#include <stdint.h>
#include <memory>

namespace guetzli {

// Returns non-zero if and only if x has a zero byte, i.e. one of
// x & 0xff, x & 0xff00, ..., x & 0xff00000000000000 is zero.
inline uint64_t HasZeroByte(uint64_t x) {
  return (x - 0x0101010101010101ULL) & ~x & 0x8080808080808080ULL;
}

// Handles the packing of bits into output bytes.
struct BitWriter {
  explicit BitWriter(size_t length) : len(length),
                                      data(new uint8_t[len]),
                                      pos(0),
                                      put_buffer(0),
                                      put_bits(64),
                                      overflow(false) {}

  void WriteBits(int nbits, uint64_t bits) {
    put_bits -= nbits;
    put_buffer |= (bits << put_bits);
    if (put_bits <= 16) {
      // At this point we are ready to emit the most significant 6 bytes of
      // put_buffer_ to the output.
      // The JPEG format requires that after every 0xff byte in the entropy
      // coded section, there is a zero byte, therefore we first check if any of
      // the 6 most significant bytes of put_buffer_ is 0xff.
      if (HasZeroByte(~put_buffer | 0xffff)) {
        // We have a 0xff byte somewhere, examine each byte and append a zero
        // byte if necessary.
        EmitByte((put_buffer >> 56) & 0xff);
        EmitByte((put_buffer >> 48) & 0xff);
        EmitByte((put_buffer >> 40) & 0xff);
        EmitByte((put_buffer >> 32) & 0xff);
        EmitByte((put_buffer >> 24) & 0xff);
        EmitByte((put_buffer >> 16) & 0xff);
      } else if (pos + 6 < len) {
        // We don't have any 0xff bytes, output all 6 bytes without checking.
        data[pos] = (put_buffer >> 56) & 0xff;
        data[pos + 1] = (put_buffer >> 48) & 0xff;
        data[pos + 2] = (put_buffer >> 40) & 0xff;
        data[pos + 3] = (put_buffer >> 32) & 0xff;
        data[pos + 4] = (put_buffer >> 24) & 0xff;
        data[pos + 5] = (put_buffer >> 16) & 0xff;
        pos += 6;
      } else {
        overflow = true;
      }
      put_buffer <<= 48;
      put_bits += 48;
    }
  }

  // Writes the given byte to the output, writes an extra zero if byte is 0xff.
  void EmitByte(int byte) {
    if (pos < len) {
      data[pos++] = byte;
    } else {
      overflow = true;
    }
    if (byte == 0xff) {
      EmitByte(0);
    }
  }

  void JumpToByteBoundary() {
    while (put_bits <= 56) {
      int c = (put_buffer >> 56) & 0xff;
      EmitByte(c);
      put_buffer <<= 8;
      put_bits += 8;
    }
    if (put_bits < 64) {
      int padmask = 0xff >> (64 - put_bits);
      int c = ((put_buffer >> 56) & ~padmask) | padmask;
      EmitByte(c);
    }
    put_buffer = 0;
    put_bits = 64;
  }

  size_t len;
  std::unique_ptr<uint8_t[]> data;
  size_t pos;
  uint64_t put_buffer;
  int put_bits;
  bool overflow;
};

}  // namespace guetzli

#endif  // GUETZLI_JPEG_BIT_WRITER_H_
