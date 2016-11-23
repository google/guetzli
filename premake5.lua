workspace "guetzli"
  configurations { "Release", "Debug" }

  flags { "C++11" }

  filter "configurations:Debug"
    symbols "On"
  filter "configurations:Release"
    optimize "Full"

  project "guetzli"
    kind "ConsoleApp"
    language "C++"
    nuget { "gflags:2.1.2.1", "libpng:1.6.26.1" }
    includedirs { ".", "third_party/butteraugli" }
    linkoptions { "-lpng", "-lgflags_nothreads" }
    -- This should work with gflags 2.x. The gflags namespace is absent in
    -- gflags-2.0, which is the version in Ubuntu Trusty package repository.
    defines { "GFLAGS_NAMESPACE=google" }
    files
      {
        "guetzli/*.cc",
        "guetzli/*.h",
        "third_party/butteraugli/butteraugli/butteraugli.cc",
        "third_party/butteraugli/butteraugli/butteraugli.h"
      }
