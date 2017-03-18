workspace "guetzli"
  configurations { "Release", "Debug" }
  filter "action:vs*"
    platforms { "x86_64", "x86" }

  flags { "C++11" }

  -- workaround for #41
  filter "action:gmake"
    symbols "On"

  filter "configurations:Debug"
    symbols "On"
  filter "configurations:Release"
    optimize "Full"

  project "guetzli"
    kind "ConsoleApp"
    language "C++"
    flags "C++11"
    includedirs { ".", "third_party/butteraugli" }
    filter "action:gmake"
      linkoptions { "`pkg-config --silence-errors --libs libpng libgflags || pkg-config --libs libpng gflags`" }
      buildoptions { "`pkg-config --silence-errors --cflags libpng libgflags || pkg-config --cflags libpng gflags`" }
    filter "action:vs*"
      links { "shlwapi" }
    filter "platforms:x86"
      architecture "x86"
    filter "platforms:x86_64"
      architecture "x86_64"
    filter {}
    files
      {
        "guetzli/*.cc",
        "guetzli/*.h",
        "third_party/butteraugli/butteraugli/butteraugli.cc",
        "third_party/butteraugli/butteraugli/butteraugli.h"
      }
