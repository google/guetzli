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
			wget https://github.com/bazelbuild/bazel/releases/download/1.2.1/bazel_1.2.1-linux-x86_64.deb
			echo '4bbb2718d451db5922223fda3aed3810c4ca50f431481e86a7fec4c585f18b1f  bazel_1.2.1-linux-x86_64.deb' | sha256sum -c --strict || exit 1
			sudo dpkg -i bazel_1.2.1-linux-x86_64.deb
			;;
		    "osx")
			brew cask install homebrew/cask-versions/adoptopenjdk8
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
