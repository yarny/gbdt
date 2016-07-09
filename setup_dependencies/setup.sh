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
    if [[ -e /usr/lib/libtcmalloc.a ]]; then
	return
    fi
    WORK_DIR=$PWD
    cd /tmp
    git clone https://github.com/gperftools/gperftools.git
    cd gperftools
    ./autogen.sh
    ./configure --prefix=$1 --enable-static --enable-shared=no
    make
    sudo make install
    cd $WORK_DIR
}

function InstallGlog()  {
    if [[ -e /usr/lib/libglog.a ]]; then
	return
    fi
    WORK_DIR=$PWD
    cd /tmp
    git clone git://github.com/google/glog
    cd glog
    mkdir -p build && cd build
    export CXXFLAGS="-fPIC" && cmake -DCMAKE_INSTALL_PREFIX=$1 .. && make VERBOSE=1
    sudo make install
    cd $WORK_DIR
}

function InstallGflags() {
    if [[ -e /usr/lib/libgflags.a ]]; then
	return
    fi
    WORK_DIR=$PWD
    cd /tmp
    git clone https://github.com/gflags/gflags.git
    cd gflags
    mkdir -p build && cd build
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
    wget https://github.com/google/protobuf/archive/v3.0.0-beta-3.1.tar.gz
    tar zxvf v3.0.0-beta-3.1.tar.gz

    cd protobuf-3.0.0-beta-3.1
    ./autogen.sh
    ./configure --disable-shared --prefix=$1
    export CXXFLAGS="-fPIC"
    make && sudo make install
    cd $WORD_DIR
}

SYSTEM_TYPE=$(SystemType)

if [[ $SYSTEM_TYPE =~ "OSX" ]];
then
    sudo brew install protobuf --devel
    sudo brew install gflags
    sudo brew install glog
    InstallGPerf "/usr/local"
else
    InstallProtoBuf3 "/usr"
    InstallGflags "/usr"
    InstallGlog "/usr"
    InstallGPerf "/usr"
fi
