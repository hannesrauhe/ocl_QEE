//Copyright 2012 Hannes Rauhe(SAP AG)

#ifndef SRC_QUERYKERNEL_H_
#define SRC_QUERYKERNEL_H_

#include <map>
#include <memory>
#include "jv2_cl/AbstractKernel.h"

struct restype_q3 {
    cl_long v0;
    unsigned k0;
};

class QueryKernel : public AbstractKernel {
  protected:
    uint data_size;
    uint total_byte_size;
    bool compiler_opt;
    int num_threads;

    std::map<int, std::shared_ptr<cl::Buffer> > cl_columns;
    std::map<std::pair<int,int>, std::pair< std::shared_ptr<cl::Buffer>, std::shared_ptr<cl::Buffer> > > cl_join_indexes;

    std::vector<std::shared_ptr<cl::Buffer> > cl_inputs;
    cl::Buffer                    cl_output;

    cl::Buffer null_buffer;
    cl::Kernel kernel;
    cl::Kernel reduce_kernel;

    unsigned arg_counter;

  public:
    QueryKernel(unsigned dev_type=CL_DEVICE_TYPE_GPU);
    virtual ~QueryKernel();

    void set_opt(bool c);

    void set_num_threads(int nt);

    void init_kernels(const char* filename, int result_size, const char* add_options = NULL);

    void ptr_as_kernel_arg(const void* array_ptr, unsigned long size, int c_id);

    void ptr_as_kernel_arg(const void* array_ptr, unsigned long size1, int c_id1, const void* offset_ptr, unsigned long size2, int c_id2);

    template <class T>
    void copy_data_to_device(
            const T* data,
            size_t size);

    void clear_data_from_device();

    template <class T>
    void exec_kernel(
            T* result,
            int result_size,
            int number_of_workgroups = 100
        );

    void synchronize();

    void reset();
};

#endif /* SRC_QUERYKERNEL_H_ */
