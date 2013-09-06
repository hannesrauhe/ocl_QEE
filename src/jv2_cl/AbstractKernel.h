/*
 * AbstractKernel.h
 *
 *  Created on: Feb 20, 2012
 *      Author: d053398
 */

#ifndef ABSTRACTKERNEL_H_
#define ABSTRACTKERNEL_H_

#define __CL_ENABLE_EXCEPTIONS

#include "CL/cl.hpp"
#include <iostream>
#include <vector>

#include <stdio.h>

class AbstractKernel {
  public:
    AbstractKernel(unsigned dev_type=CL_DEVICE_TYPE_GPU, bool verbose=true);
    virtual ~AbstractKernel() {}

  protected:
    std::vector<std::pair<const char*,double> > exectimes;
    cl::Context      context;
    cl::CommandQueue queue;
    cl::Program      program;

  public:
    std::vector<cl::Device> devices;
    unsigned long           global_mem_size;
    int                     local_mem_size;
    int                     max_alloc_size;
    int                     max_work_group_size;
    cl_int err;

    void load_program(const char* kernel_source, const char* build_options="");
    void push_exec_time(const char* name, const cl::Event& event);
    double get_exec_time_by_category(const char* cat = NULL) const;
    void print_exec_times(const char* cat = NULL) const;
};

#endif /* ABSTRACTKERNEL_H_ */
