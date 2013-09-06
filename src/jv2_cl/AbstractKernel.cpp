//Copyright 2012 <Hannes Rauhe (SAP AG)>
#include "jv2_cl/AbstractKernel.h"
#include <fstream>
#include <cstring>

AbstractKernel::AbstractKernel(unsigned dev_type, bool verbose) {
    bool found_device = false;
    printf("Initialize OpenCL object and context\n");

    std::vector<cl::Platform> platforms;
    err = cl::Platform::get(&platforms);
    printf("number of platforms: %u\n", (unsigned)platforms.size());

    cl::Device device;

    for (std::vector<cl::Platform>::iterator it = platforms.begin(); it < platforms.end(); ++it) {
        std::cout<<" "<<it->getInfo<CL_PLATFORM_VERSION>()<<std::endl;
        std::cout<<" "<<it->getInfo<CL_PLATFORM_NAME>()<<std::endl;
//        std::cout<<" "<<it->getInfo<CL_PLATFORM_VENDOR>()<<std::endl;
//        std::cout<<" "<<it->getInfo<CL_PLATFORM_PROFILE>()<<std::endl;
//        std::cout<<" "<<it->getInfo<CL_PLATFORM_EXTENSIONS>()<<std::endl;


        try {
            cl_context_properties properties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties) (*it)(), 0 };
            context = cl::Context(dev_type, properties);
            devices = context.getInfo<CL_CONTEXT_DEVICES>();
            printf(" number of devices: %u\n", (unsigned)devices.size());
            if (devices.size() && !found_device) {
                device = devices[0];
                found_device = true;
                printf("--> Using the first device of this platform!");
            }
        } catch (const cl::Error& er) {
            printf("ERROR: %s(%d)\n", er.what(), er.err());
        }

        printf("\n");
    }

    if(!found_device) {
        throw "No device found";
    }

    global_mem_size     = device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
    local_mem_size      = device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
    max_alloc_size      = device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
    max_work_group_size = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    uint reads  = device.getInfo<CL_DEVICE_MAX_READ_IMAGE_ARGS>();
    uint writes = device.getInfo<CL_DEVICE_MAX_WRITE_IMAGE_ARGS>();
    uint units  = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
    if(verbose) {
        printf("Global Memory Size: %lud (%lud KB/%lud MB)\n",
                global_mem_size,
                global_mem_size / 1024,
                global_mem_size / 1024 / 1024);
        printf("Local Memory Size: %d (%d KB/%d MB)\n", local_mem_size, local_mem_size / 1024, local_mem_size / 1024 / 1024);
        printf("Available for allocation: %d (%d KB/%d MB)\n",
                max_alloc_size,
                max_alloc_size / 1024,
                max_alloc_size / 1024 / 1024);
        printf("Maximum work group size: %d \n", max_work_group_size);
        printf("Maximum read arg count: %d \n", reads);
        printf("Maximum write arg count: %d \n", writes);
        printf("Compute units: %d\n", units);
    }

    try {
        queue = cl::CommandQueue(context, devices[0],
                    CL_QUEUE_PROFILING_ENABLE, &err);
    } catch (cl::Error& er) {
        printf("ERROR: %s(%d)\n", er.what(), er.err());
        throw "Error while Creating CL-Queue";
    }
}

void AbstractKernel::load_program(const char* file_name, const char* build_options) {
    char* kernel_source;
#ifdef _DEBUG
    printf("load the program\n");
#endif
    std::ifstream cl_file(file_name);
    if (cl_file.is_open()) {
        int length;      // open input file
        cl_file.seekg(0, std::ios::end);    // go to the end
        length = cl_file.tellg();           // report location (this is the length)
        cl_file.seekg(0, std::ios::beg);    // go back to the beginning
        kernel_source = (char*)malloc(length * sizeof(char));    // allocate memory for a buffer of appropriate dimension
        cl_file.read(kernel_source, length);       // read the whole file into the buffer
        cl_file.close();
        try {
            cl::Program::Sources source(1, std::make_pair(kernel_source, length));
            program = cl::Program(context, source);
        } catch (const cl::Error& er) {
            printf("ERROR: %s(%d)\n", er.what(), er.err());
            throw "Error while loading CL program";
        }
#ifdef _DEBUG
        printf("build program\n");
#endif
        try {
            program.build(devices,build_options);
        } catch (const cl::Error& er) {
            printf("BUILD ERROR: %s(%d)\n", er.what(), er.err());
            std::string log;
            program.getBuildInfo(devices[0], CL_PROGRAM_BUILD_LOG, &log);
            cl_build_status status = program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(devices[0]);
            printf("Error %d - ", status);
            printf("%s", log.c_str());
            printf("\n");
            throw "Error while compiling CL c-code";
        }
#ifdef _DEBUG
        printf("done building program\n");
#endif
    } else {
        printf("%s cannot be opened\n",file_name);
        throw "CL c-code not found";
    }
}

void AbstractKernel::push_exec_time(const char* name, const cl::Event& event) {
//    event.wait();
    long eventStart = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    long eventEnd = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
    double d_seconds = 1.0e-9 * (double)(eventEnd - eventStart);
    exectimes.push_back(std::make_pair(name,d_seconds));
}

double AbstractKernel::get_exec_time_by_category(const char* cat) const {
    double ret = 0.0;
    for(auto it = exectimes.begin(); it!=exectimes.end(); ++it) {
        if(cat==NULL || strstr(it->first,cat)==it->first) {
            ret+=it->second;
        }
    }
    return ret;
}

void AbstractKernel::print_exec_times(const char* cat) const {
    for(auto it = exectimes.begin(); it!=exectimes.end(); ++it) {
        if(cat==NULL || strstr(it->first,cat)==it->first) {
            printf("%s\t%f\n",it->first,it->second);
        }
    }
}
