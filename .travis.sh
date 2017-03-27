#!/bin/bash

case "$1" in
    "install")
	case "${TRAVIS_OS_NAME}" in
	    "linux")
		wget https://github.com/bazelbuild/bazel/releases/download/0.4.5/bazel_0.4.5-linux-x86_64.deb
		echo 'b494d0a413e4703b6cd5312403bea4d92246d6425b3be68c9bfbeb8cc4db8a55  bazel_0.4.5-linux-x86_64.deb' | sha256sum -c --strict || exit 1
		sudo dpkg -i bazel_0.4.5-linux-x86_64.deb
		;;
	    "osx")
		brew update
		brew install binutils
		brew install bazel
		;;
	esac
	;;
    "script")
	case "${BUILD_SYSTEM}" in
	    "bazel")
		bazel test -c opt ...:all
		;;
	esac
	;;
esac
