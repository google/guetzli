set(sources
  butteraugli_comparator.cc
  dct_double.cc
  debug_print.cc
  entropy_encode.cc
  fdct.cc
  gamma_correct.cc
  guetzli.cc
  idct.cc
  jpeg_data.cc
  jpeg_data_decoder.cc
  jpeg_data_encoder.cc
  jpeg_data_reader.cc
  jpeg_data_writer.cc
  jpeg_huffman_decode.cc
  output_image.cc
  preprocess_downsample.cc
  processor.cc
  quality.cc
  quantize.cc
  score.cc)

add_executable(guetzli ${sources})
target_link_libraries(guetzli ${PNG_LIBRARY} butteraugli)

add_library(guetzli_static STATIC ${sources})
target_link_libraries(guetzli ${PNG_LIBRARY} butteraugli)
