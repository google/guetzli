#include "guetzli/processor.h"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  std::string jpeg_data(reinterpret_cast<const char*>(data), size);
  std::string jpeg_out;

  // TODO(robryk): Use nondefault parameters.
  guetzli::Params params;
  (void)guetzli::Process(params, nullptr, jpeg_data, &jpeg_out);
  // TODO(robryk): Verify output distance if Process() succeeded.
  return 0;
}




