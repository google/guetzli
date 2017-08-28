# Changelog

## Unreleased

Fixes:

 * Fixed various warnings on MSVC. (#60)
 * Fixed an occurrence of arithmetic overflow.
 * Fixed a bug in command-line argument parsing.

Improvements:

 * Memory usage has been reduced. (#148, #156, #213)
 * A fuzzing target has been added.
 * `--` can now be used to stop parsing command-line arguments.
 * Builds are now tested on Travis CI and Appveyor. (#136, #191)
 * Improved error messages.
 * Improved build instructions.

## v1.0.1

Released 2017-03-21.

Fixes:

 * Fixed yet another crash for small images. (#38)
 * Makes metadata handling consistent. (#100)
 * Fixed some incorrect error messages. (#40)
 * Fixed various build issues (a.o. removes the gflags dependency).
 * Add missing documentation.

Improvements:

 * Adds memory limit support. (#75)

## v1.0

Released 2017-03-15.

Fixes:

 * Fixed two crashes. (#29 and 5de9ad8)
 * Made memory requirements more explicit.

## v0.2

Released 2016-01-17.

Fixes:

 * The same filename can now be used for input and output. (#20)
 * Various compilation issues were fixed. (#12, #14)
 * Fixed a crash on Windows due to usage of uninitialized memory. (#23)

Improvements:

 * 64-bit binaries for Windows are now released.

## v0.1

Released 2016-12-21.

Fixes:

 * Fixed generated jpegs which some viewers were unable to open. (#1)
 * Fixed the build system for Windows.

## v0

Released 2016-10-21.

Initial release.
