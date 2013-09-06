#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int64_base_atomics : enable

#define RESULT_SIZE 5
__kernel void local_sum(__local long* result,const long p) {
    __local long l_sum[NUM_OF_THREADS];
    size_t local_id = get_local_id(0);
    l_sum[local_id] = p;

    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id < 256) {
        l_sum[local_id] += l_sum[local_id+256];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id < 128) {
        l_sum[local_id] += l_sum[local_id+128];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id < 64) {
        l_sum[local_id] += l_sum[local_id+64];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id < 32) {
        l_sum[local_id] += l_sum[local_id+32];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id < 16) {
        l_sum[local_id] += l_sum[local_id+16];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id < 8) {
        l_sum[local_id] += l_sum[local_id+8];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id < 4) {
        l_sum[local_id] += l_sum[local_id+4];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id < 2) {
        l_sum[local_id] += l_sum[local_id+2];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id < 1) {
        *result += l_sum[local_id] + l_sum[local_id+1];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

//not right
__kernel void q(
        __global unsigned short* p_c1,
        __global unsigned*     p_c0,
        __global unsigned*    mo_t0_to_t1,
        __global unsigned short* p_c2 ,
        __global unsigned short* p_c3,
        unsigned gend,
        unsigned rounds_per_thread,
        __global long* globalResult) {
    __local long l_thread_data[RESULT_SIZE];

    for(int i = get_local_id(0);i<RESULT_SIZE;i+=get_local_size(0)) {
        l_thread_data[i] = (long)(0);
    }

    unsigned lstart = (get_group_id(0)*get_local_size(0)*rounds_per_thread)+get_local_id(0);
    unsigned lend = lstart + get_local_size(0)*rounds_per_thread;
    if(lend>gend) lend=gend;

    for(unsigned i0=lstart;i0<lend;i0+=get_local_size(0))
    {
        unsigned short c1 = p_c1[i0];
        if (!(c1 >= 8582))
            continue;
        if (!(c1 < 8674))
            continue;
        unsigned c0 = p_c0[i0];
        unsigned rkey    = c0;
        unsigned mo1rend = mo_t0_to_t1[i0+1];
        for (unsigned i1 = mo_t0_to_t1[i0]; i1 < mo1rend; ++i1) {
            unsigned short c2 = p_c2[i1];
            unsigned short c3 = p_c3[i1];
            if (!(c2 < c3))
                continue;
            atomic_inc(&l_thread_data[rkey]);
            break;
        }
    }


    for(int i = get_local_id(0);i<RESULT_SIZE;i+=get_local_size(0)) {
        unsigned result_pos = i + get_group_id(0)*RESULT_SIZE;
        globalResult[result_pos]=l_thread_data[i];
    }
}

__kernel void q_reduce(
        __global long* interimResult,
        __global long* globalResult,
        unsigned noOfInterimResults) {
    __local long l_thread_data[RESULT_SIZE];
    for(int i = get_local_id(0);i<RESULT_SIZE;i+=get_local_size(0)) {
        l_thread_data[i] = (long)(0);
    }
    unsigned lim = noOfInterimResults/get_local_size(0);
    lim = get_local_size(0)*(lim+1);

    for(unsigned resNumber = get_local_id(0);resNumber<lim;resNumber+=get_local_size(0)) {
        for(unsigned lineNumber=0;lineNumber<RESULT_SIZE;++lineNumber) {
            long a=interimResult[resNumber*RESULT_SIZE+lineNumber]*(resNumber<noOfInterimResults);
            local_sum(&l_thread_data[lineNumber],a);
        }
    }

    for(unsigned short result_pos = get_local_id(0);result_pos<RESULT_SIZE;result_pos+=get_local_size(0)) {
        globalResult[result_pos]=l_thread_data[result_pos];
    }
}
