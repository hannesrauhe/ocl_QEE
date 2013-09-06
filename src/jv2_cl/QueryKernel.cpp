//Copyright 2012 Hannes Rauhe (SAP AG)

#include "jv2_cl/QueryKernel.h"
#include <malloc.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "sys/time.h"

#include "timer.hpp"

QueryKernel::QueryKernel(unsigned dev_type) :
        AbstractKernel(dev_type),data_size(0),total_byte_size(0),compiler_opt(true),num_threads(512) {
    set_num_threads(512);
}

QueryKernel::~QueryKernel() {}

void QueryKernel::set_opt(bool c) {
    compiler_opt = c;
}

void QueryKernel::set_num_threads(int nt) {
    num_threads = nt;
    if(max_work_group_size<num_threads) {
        num_threads = max_work_group_size;
        printf("changed number of threads to %d\n",num_threads);
    }
    if(!(num_threads==256 || num_threads==512)) {
        throw "Sorry, only 256/512 threads per workgroup allowed atm";
    }
}

void QueryKernel::init_kernels(const char* filename, int result_size, const char* add_options) {
    assert(result_size>0);
    CTimer timer1("Load Code & Compilation");
    {
        char options[255];
        sprintf(options,"-DRESULT_SIZE=%d -DNUM_OF_THREADS=%d",result_size,num_threads);
        if(!compiler_opt) {
            strcat(options," -cl-opt-disable");
        }
        if(add_options!=NULL) {
            strcat(options," ");
            strcat(options,add_options);
        }
        printf("%s\n",options);
        load_program(filename,options);
    }
    printf("initializing kernels from %s\n",filename);

    //initialize our kernel from the program
    try {
        kernel        = cl::Kernel(program, "q", &err);
        reduce_kernel = cl::Kernel(program, "q_reduce", &err);
        arg_counter   = 0;
    } catch (const cl::Error& er) {
        printf("ERROR: %s(%d)\n", er.what(), er.err());
        throw "error while initializing kernel";
    }
}


//#define USE_MAP

void QueryKernel::ptr_as_kernel_arg(const void* array_ptr, unsigned long size, int c_id) {
    try {
        if(cl_columns.find(c_id)==cl_columns.end()) {
#ifndef USE_MAP
            std::shared_ptr<cl::Buffer> p(new cl::Buffer(context, CL_MEM_READ_ONLY, size , NULL, &err));
#endif
            cl_columns[c_id]=p;
#ifndef USE_MAP
            cl::Event              event;
            queue.enqueueWriteBuffer(*cl_columns[c_id], CL_TRUE, 0, size, array_ptr, NULL, &event);
            event.wait();
            push_exec_time("transferColumn",event);
#endif
            total_byte_size+=size;
        }
        cl_inputs.push_back(cl_columns[c_id]);
        kernel.setArg(arg_counter++, *(cl_inputs.back()));
    } catch (const cl::Error& er) {
        printf("ERROR: %s(%d)\n...with size %lu", er.what(), er.err(), size);
        return;
    }
}

void QueryKernel::ptr_as_kernel_arg(const void* array_ptr, unsigned long size1, int c_id1, const void* offset_ptr, unsigned long size2, int c_id2) {
    std::pair<int,int> keyp(c_id1,c_id2);
    try {
        if(cl_join_indexes.find(keyp)==cl_join_indexes.end()) {
#ifndef USE_MAP
            std::shared_ptr<cl::Buffer> p1(new cl::Buffer(context, CL_MEM_READ_ONLY, size1 , NULL, &err));
#endif
            std::shared_ptr<cl::Buffer> p2;
            if(size2) {
#ifndef USE_MAP
                p2 = std::shared_ptr<cl::Buffer>(new cl::Buffer(context, CL_MEM_READ_ONLY, size2 , NULL, &err));
#endif
            }
            cl_join_indexes[keyp]=std::make_pair(p1,p2);
#ifndef USE_MAP
            cl::Event              event;
            queue.enqueueWriteBuffer(*(cl_join_indexes[keyp].first), CL_TRUE, 0, size1, array_ptr, NULL, &event);
            if(size2) {
                queue.enqueueWriteBuffer(*(cl_join_indexes[keyp].second), CL_TRUE, 0, size2, offset_ptr, NULL, &event);
            }
            event.wait();
            push_exec_time("transferIndex",event);
#endif
        }
        //the offset first
        if(size2) {
            cl_inputs.push_back(cl_join_indexes[keyp].second);
            kernel.setArg(arg_counter++, *(cl_inputs.back()));
        }
        //then the array
        cl_inputs.push_back(cl_join_indexes[keyp].first);
        kernel.setArg(arg_counter++, *(cl_inputs.back()));

        total_byte_size+=size1+size2;
    } catch (const cl::Error& er) {
        printf("ERROR: %s(%d)\n...with size %lu (array) and size %lu (offset)", er.what(), er.err(), size1, size2);
        throw "Error while Loading Index";
        return;
    }

}

void QueryKernel::clear_data_from_device() {
    cl_join_indexes.clear();
    cl_columns.clear();
}

template<class T>
void QueryKernel::exec_kernel(
        T* result,
        int result_size,
        int number_of_workgroups
    ) {
    int number_of_threads=num_threads;
    printf("Data size processed by kernel: %u\n",total_byte_size);
    CTimer timer4("Executing Kernel");
    const unsigned              rounds_per_thread   = data_size/number_of_threads/number_of_workgroups+1;
    const unsigned              interim_size      = number_of_workgroups*result_size;


    cl::Buffer cl_interim = cl::Buffer(context, CL_MEM_READ_WRITE, interim_size * sizeof(T), NULL, &err);
    cl::Buffer cl_output  = cl::Buffer(context, CL_MEM_READ_WRITE, result_size * sizeof(T), NULL, &err);

#ifdef _DEBUG
    try {
        void* initmem = calloc(interim_size,sizeof(T));
        if(initmem == NULL) {
            throw "Error while initializing main(!) memory (zeros)";
        }
        cl::Event              event;
        queue.enqueueWriteBuffer(cl_interim, CL_TRUE, 0, interim_size * sizeof(T), initmem, NULL, &event);
        queue.enqueueWriteBuffer(cl_output, CL_TRUE, 0, result_size * sizeof(T), initmem, NULL, &event);
        event.wait();
        free(initmem);
    } catch (cl::Error& er) {
        printf("ERROR: %s(%d)\n", er.what(), er.err());
        throw "Error while initializing device memory (zeros)";
    }
#endif

    try {
        cl::Event              event;
        kernel.setArg(arg_counter++, data_size);
        kernel.setArg(arg_counter++, rounds_per_thread);
        kernel.setArg(arg_counter++, cl_interim);
        std::cout<<"Executing Kernel on "<<data_size<<" rows with "<<
        (unsigned)(number_of_workgroups*number_of_threads)<<" threads in "<<number_of_workgroups<<" workgroups"<<std::endl;
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(
                        number_of_workgroups*number_of_threads), number_of_threads, NULL, &event);
        event.wait();
        push_exec_time("kernelExec",event);
    } catch (cl::Error& er) {
        printf("ERROR: %s(%d)\n", er.what(), er.err());
        throw "Error while executing main kernel";
    }

#ifdef _DEBUG
    try{
        cl::Event event;
        T* res = (T*) malloc(interim_size * sizeof(T));
        err = queue.enqueueReadBuffer(cl_interim, CL_TRUE, 0, interim_size * sizeof(T), res, NULL, &event);
        for( int i = 0;i<interim_size;++i ) {
            //TODO warning! casting because of laziness
            if(i%result_size==0) {
                printf("___\n");
            }
            if(reinterpret_cast<long*>(res)[i]!=0)
                printf("%ld\n",res[i]);
        }
        free(res);
    } catch (cl::Error& er) {
        printf("ERROR: %s(%d)\n", er.what(), er.err());
        throw "Error while executing main kernel";
    }
#endif

    try {
        cl::Event              event;
        reduce_kernel.setArg(0, cl_interim);
        reduce_kernel.setArg(1, cl_output);
        reduce_kernel.setArg(2, number_of_workgroups);
//        int num_threads = std::min(number_of_workgroups, number_of_threads);
//        num_threads = std::max(num_threads,result_size);
        std::cout<<"Executing Reduce Kernel"<<std::endl; // on "<<data_size<<" rows with "<<(unsigned)(num_work_groups*512)<<" threads in "<<num_work_groups<<" workgroups"<<std::endl;
        queue.enqueueNDRangeKernel(reduce_kernel, cl::NullRange, cl::NDRange(
                number_of_threads), number_of_threads /*cl::NullRange*/, NULL, &event);
        event.wait();
        push_exec_time("kernelReduce",event);
    } catch (cl::Error& er) {
        printf("ERROR: %s(%d)\n", er.what(), er.err());
        throw "Error while executing reduce kernel";
    }

    try {
        cl::Event              event;
        err = queue.enqueueReadBuffer(cl_output, CL_TRUE, 0, result_size * sizeof(T), result, NULL, &event);
        event.wait();
        push_exec_time("transferResult",event);
    }  catch (cl::Error& er) {
        printf("ERROR: %s(%d)\n", er.what(), er.err());
        throw "Error while receiving result";
    }
}

template void QueryKernel::exec_kernel(cl_long8*, int, int);
template void QueryKernel::exec_kernel(cl_long4*, int, int);
template void QueryKernel::exec_kernel(cl_long2*, int, int);
template void QueryKernel::exec_kernel(cl_long*, int, int);
template void QueryKernel::exec_kernel(restype_q3*, int, int);

void QueryKernel::reset() {
//    try {
//        queue.finish();
//    } catch (cl::Error& er) {
//        printf("ERROR: %s(%d)\n", er.what(), er.err());
//        throw "Queue cannot be finished while reseting";
//    }
    cl_inputs.clear();
    exectimes.clear();
    arg_counter=0;
    data_size=0;
    total_byte_size=0;
}

void QueryKernel::synchronize() {
    try {
        queue.finish();
    } catch (const cl::Error& er) {
        printf("ERROR: %s(%d)\n", er.what(), er.err());
        throw "Error while synchronizing";
    }
}
