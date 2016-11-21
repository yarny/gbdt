#!/bin/bash

# TODO(criver): Swtich installation to /usr/local when bazel is able to deal with /usr/local.
function SystemType() {
    if [[ "${OSTYPE}" =~ "darwin"* ]];
    then
	echo "OSX"
	return
    elif [[ $(lsb_release -r) =~ "14" ]];
    then
	echo "Ubuntu14"
	return
    elif [[ $(lsb_release -r) =~ "12" ]];
    then
	echo "Ubuntu12"
	return
    else
	echo "Unknown"
    fi
}

function InstallGPerf()  {
    if [[ -e $1/lib/libtcmalloc.a ]]; then
	return
    fi
    WORK_DIR=$PWD
    cd /tmp
    git clone https://github.com/gperftools/gperftools.git
    cd gperftools
    ./autogen.sh
    ./configure --prefix=$1 --enable-static --disable-shared
    export CXXFLAGS="-fPIC" && make && sudo make install
    cd $WORK_DIR
}

function InstallGlog()  {
    if [[ -e $1/lib/libglog.a ]]; then
	return
    fi
    WORK_DIR=$PWD
    cd /tmp
    git clone git://github.com/google/glog
    cd glog
    mkdir -p build_targets && cd build_targets
    export CXXFLAGS="-fPIC" && cmake -DCMAKE_INSTALL_PREFIX=$1 .. && make VERBOSE=1
    sudo make install
    cd $WORK_DIR
}

function InstallGflags() {
    if [[ -e $1/lib/libgflags.a ]]; then
	return
    fi
    WORK_DIR=$PWD
    cd /tmp
    git clone https://github.com/gflags/gflags.git
    cd gflags
    mkdir -p build_targets && cd build_targets
    export CXXFLAGS="-fPIC"
    cmake -DCMAKE_INSTALL_PREFIX=$1 -DGFLAGS_NAMESPACE=google ..
    make && sudo make install
    cd $WORK_DIR
}

function InstallProtoBuf3() {
    PROTO_VERSION=`protoc --version`
    if [[ $PROTO_VERSION =~ "3.0.0" ]]; then
        return
    fi
    WORK_DIR=$PWD

    cd /tmp
    git clone https://github.com/google/protobuf.git
    cd protobuf

    ./autogen.sh
    ./configure --prefix=$1 --disable-shared --with-pic
    export CXXFLAGS="-fPIC" && make && sudo make install
    cd $WORK_DIR
}

SYSTEM_TYPE=$(SystemType)

if [[ $SYSTEM_TYPE =~ "OSX" ]];
then
    sudo brew install google-perftools
    sudo brew link --overwrite gperftools
    sudo rm /usr/loca/lib/libtcmalloc*dylib*

    InstallProtoBuf3 "/usr/local"
    InstallGflags "/usr/local"
    InstallGlog "/usr/local"
else
    InstallGPerf "/usr"
    InstallProtoBuf3 "/usr"
    InstallGflags "/usr"
    InstallGlog "/usr"
fi
