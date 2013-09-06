TARGET = main
CXX = g++-4.7
CXXFLAGS += -O3 -Wall --std=c++0x

CPP_FILES = src/jv2_opencl.cpp src/jv2_cl/AbstractKernel.cpp
O_FILES = $(patsubst %.cpp,%.o,$(CPP_FILES))

BOOST_PATH = /opt/build-essentials
TBB_PATH = /opt/build-essentials

INCL = -Isrc -I$(BOOST_PATH)/include
LIBS = -lOpenCL -ltbbmalloc -ltbb -lboost_serialization -lboost_thread -lbz2 -lboost_date_time -lboost_chrono -lboost_iostreams -lboost_system
LIBPATH = -L$(BOOST_PATH)/lib -L$(TBB_PATH)/lib

all: client jv2_opencl

client: src/client.cpp
	$(CXX) src/client.cpp -o client

jv2_opencl: jv2_opencl.o QueryKernel.o AbstractKernel.o tpch_data_load.o
	$(CXX) $(LIBPATH) -fopenmp -o jv2_opencl jv2_opencl.o QueryKernel.o AbstractKernel.o tpch_data_load.o $(LIBS) # -lbotan

jv2_opencl.o: src/jv2_opencl.cpp src/jv2_cl/tpch_kernel.h
	$(CXX) $(INCL) $(CXXFLAGS) -c src/jv2_opencl.cpp

QueryKernel.o: src/jv2_cl/QueryKernel.cpp src/jv2_cl/QueryKernel.h src/jv2_cl/timer.hpp
	$(CXX) $(INCL) $(CXXFLAGS) -c src/jv2_cl/QueryKernel.cpp

AbstractKernel.o: src/jv2_cl/AbstractKernel.cpp src/jv2_cl/AbstractKernel.h
	$(CXX) $(INCL) $(CXXFLAGS) -c src/jv2_cl/AbstractKernel.cpp

tpch_data_load.o: src/jv2_cl/tpch_data_load.cpp src/jv2_cl/tpch_data_load.h
	$(CXX) $(INCL) $(CXXFLAGS) -c src/jv2_cl/tpch_data_load.cpp

clean:
	rm *.o jv2_opencl client

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/tbb30_20110427oss/build/linux_intel64_gcc_cc4.5.3_libc2.11.1_kernel2.6.32_release:/opt/boost/boost_1_51_0/stage/lib
