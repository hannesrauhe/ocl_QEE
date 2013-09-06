__kernel void local_sum(__local long2* result,const long2 p) {
    __local long2 l_sum[NUM_OF_THREADS];
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
        __global const unsigned short* p_c3,
        __global const unsigned*        t0_to_t1,
        __global const int*            p_c0,
        __global const char*           p_c1,
        __global const unsigned*       p_c2,
        unsigned gend,
        unsigned rounds_per_thread,
        __global long2* globalResult) {
    __local long2 l_thread_data[RESULT_SIZE];
    long2 p_thread_data[RESULT_SIZE];

    for(int i = 0;i<RESULT_SIZE;++i) {
        p_thread_data[i] = (long2)(0);
    }
    for(int i = get_local_id(0);i<RESULT_SIZE;i+=get_local_size(0)) {
        l_thread_data[i] = (long2)(0);
    }

    unsigned lstart = (get_group_id(0)*get_local_size(0)*rounds_per_thread)+get_local_id(0);
    unsigned lend = lstart + get_local_size(0)*rounds_per_thread;
    if(lend>gend) lend=gend;

    for (unsigned i0 = lstart; i0 < lend;i0+=get_local_size(0)) {
        unsigned short c3 = p_c3[i0];
        if (!(c3 >= 9374)) continue;
        if (!(c3 < 9404)) continue;
        unsigned i1 = t0_to_t1[i0];
        int c0 = p_c0[i0];
        char c1 = p_c1[i0];
        long     agg1 = (c0*(100-c1));
        unsigned c2   = p_c2[i1];
        long agg0 = ((c0*(100-c1))*((c2 >= 75)*(c2 <= 99)));
        p_thread_data[0]+=(long2)(agg0, agg1);
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    for(int j=0;j<RESULT_SIZE;++j) {
        local_sum(&l_thread_data[j],p_thread_data[j]);
    }

    for(int i = get_local_id(0);i<RESULT_SIZE;i+=get_local_size(0)) {
        unsigned result_pos = i + get_group_id(0)*RESULT_SIZE;
        globalResult[result_pos]=l_thread_data[i];
    }
}

__kernel void q_reduce(
        __global long2* interimResult,
        __global long2* globalResult,
        unsigned noOfInterimResults) {
    __local long2 l_thread_data[RESULT_SIZE];
    for(int i = get_local_id(0);i<RESULT_SIZE;i+=get_local_size(0)) {
        l_thread_data[i] = (long2)(0);
    }
    unsigned lim = noOfInterimResults/get_local_size(0);
    lim = get_local_size(0)*(lim+1);

    for(unsigned resNumber = get_local_id(0);resNumber<lim;resNumber+=get_local_size(0)) {
        for(unsigned lineNumber=0;lineNumber<RESULT_SIZE;++lineNumber) {
            long2 a=interimResult[resNumber*RESULT_SIZE+lineNumber]*(resNumber<noOfInterimResults);
            local_sum(&l_thread_data[lineNumber],a);
        }
    }

    for(unsigned short result_pos = get_local_id(0);result_pos<RESULT_SIZE;result_pos+=get_local_size(0)) {
        globalResult[result_pos]=l_thread_data[result_pos];
    }
}

