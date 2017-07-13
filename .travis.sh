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
			wget https://github.com/bazelbuild/bazel/releases/download/0.5.2/bazel_0.5.2-linux-x86_64.deb
			echo 'b14c8773dab078d3422fe4082f3ab4d9e14f02313c3b3eb4b5b40c44ce29ed59  bazel_0.5.2-linux-x86_64.deb' | sha256sum -c --strict || exit 1
			sudo dpkg -i bazel_0.5.2-linux-x86_64.deb
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
