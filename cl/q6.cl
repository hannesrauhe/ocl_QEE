__kernel void local_sum(__local long* result,const long p) {
    __local long l_sum[NUM_OF_THREADS];
    size_t local_id = get_local_id(0);
    l_sum[local_id] = p;

#if NUM_OF_THREADS>=512
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id < 256) {
        l_sum[local_id] += l_sum[local_id+256];
    }
#endif
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

__kernel void q(
        __global const unsigned short* p_c2,
        __global const char*       p_c1,
        __global const char*        p_c3,
        __global const int*           p_c0 ,
        unsigned gend,
        unsigned rounds_per_thread,
        __global long* globalResult) {
    __local long l_thread_data[RESULT_SIZE];
    for(int i = get_local_id(0);i<RESULT_SIZE;i+=get_local_size(0)) {
        l_thread_data[i] = (long)(0);
    }

    long p_thread_data = 0;

    unsigned lstart = (get_group_id(0)*get_local_size(0)*rounds_per_thread)+get_local_id(0);
    unsigned lend = lstart + get_local_size(0)*rounds_per_thread;
    if(lend>gend) lend=gend;

    for(unsigned i0=lstart;i0<lend;i0+=get_local_size(0))
    {
        unsigned short c2 = p_c2[i0];
        if (!(c2 >= 8766))
            continue;
        char c1 = p_c1[i0];
        if (!(c1 <= 7))
            continue;
        char c3 = p_c3[i0];
        if (!(c3 < 24))
            continue;
        if (!(c2 < 9131))
            continue;
        if (!(c1 >= 5))
            continue;
        int c0 = p_c0[i0];
        p_thread_data+=(c0*c1);
    }


    local_sum(&l_thread_data,p_thread_data);

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
