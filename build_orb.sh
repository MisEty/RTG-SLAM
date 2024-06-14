#!/bin/bash  

mkdir -p thirdParty && cd thirdParty
install_path=$(pwd)/install
mkdir -p ${install_path}

python_prefix=$(python -c "import sys; print(sys.prefix)")  
python_include=${python_prefix}/include/python3.9/
python_lib=${python_prefix}/lib/libpython3.9.so
python_exe=${python_prefix}/bin/python
python_env=${python_prefix}/lib/python3.9/site-packages/
numpy_include=$(python -c "import numpy; print(numpy.get_include())")  

echo ${python_env}

# # build pangolin
git clone -b v0.5 https://github.com/stevenlovegrove/Pangolin.git
cd Pangolin
mkdir -p build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=${install_path}
make install -j

# build opencv-4.2.0
cd ../../
wget https://github.com/opencv/opencv/archive/4.2.0.zip
unzip 4.2.0.zip
cd opencv-4.2.0
mkdir -p build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=${install_path}
make install -j 

opencv_dir=${install_path}/lib/cmake/opencv4

# build orbslam2
cd ../../
cd ORB-SLAM2-PYBIND
bash build.sh ${opencv_dir} ${install_path}
cd ../


# # build pybind
# # build boost
wget -t 999 -c https://boostorg.jfrog.io/artifactory/main/release/1.80.0/source/boost_1_80_0.zip
unzip boost_1_80_0.zip
cd boost_1_80_0
./bootstrap.sh --with-libraries=python --prefix=${install_path} --with-python=${python_exe}

# # ./b2
./b2 install --with-python include=${python_include} --prefix=${install_path}


# # build orbslam_pybind
cd ../pybind
mkdir -p build && cd build

cmake .. -DPYTHON_INCLUDE_DIRS=${python_include} \
         -DPYTHON_LIBRARIES=${python_lib} \
         -DPYTHON_EXECUTABLE=${python_exe} \
         -DBoost_INCLUDE_DIRS=${install_path}/include/boost \
         -DBoost_LIBRARIES=${install_path}/lib/libboost_python39.so \
         -DORB_SLAM2_INCLUDE_DIR=${install_path}/include/ORB_SLAM2 \
         -DORB_SLAM2_LIBRARIES=${install_path}/lib/libORB_SLAM2.so \
         -DOpenCV_DIR=${install_path}/lib/cmake/opencv4 \
         -DPangolin_DIR=${install_path}/lib/cmake/Pangolin \
         -DPYTHON_NUMPY_INCLUDE_DIR=${numpy_include} \
         -DCMAKE_INSTALL_PREFIX=${python_env}

make install -j

