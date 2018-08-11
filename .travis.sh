#!/bin/bash

case "$1" in
    "install")
	case "${TRAVIS_OS_NAME}" in
	    "linux")
		;;
	    "osx")
		brew update
		brew install netpbm
		;;
	esac
	case "${BUILD_SYSTEM}" in
	    "bazel")
		case "${TRAVIS_OS_NAME}" in
		    "linux")
			sudo apt-get remove openjdk-9-jdk oracle-java9-installer # Conflicts with Bazel.
			wget https://github.com/bazelbuild/bazel/releases/download/0.4.5/bazel_0.4.5-linux-x86_64.deb
			echo 'b494d0a413e4703b6cd5312403bea4d92246d6425b3be68c9bfbeb8cc4db8a55  bazel_0.4.5-linux-x86_64.deb' | sha256sum -c --strict || exit 1
			sudo dpkg -i bazel_0.4.5-linux-x86_64.deb
			;;
		    "osx")
			brew install bazel
			;;
		esac
		;;
	    "make")
		;;
	esac
	;;
    "script")
	case "${BUILD_SYSTEM}" in
	    "bazel")
		bazel build -c opt ...:all
		GUETZLI_BIN=bazel-bin/guetzli
		;;
	    "make")
		make
		GUETZLI_BIN=bin/Release/guetzli
		;;
	esac
	tests/smoke_test.sh $GUETZLI_BIN
	;;
esac
