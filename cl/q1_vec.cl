__kernel void local_sum(__local long8* result,const long8 p) {
    __local long8 l_sum[NUM_OF_THREADS];

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
//    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id < 16) {
        l_sum[local_id] += l_sum[local_id+16];
    }
//    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id < 8) {
        l_sum[local_id] += l_sum[local_id+8];
    }
//    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id < 4) {
        l_sum[local_id] += l_sum[local_id+4];
    }
//    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id < 2) {
        l_sum[local_id] += l_sum[local_id+2];
    }
//    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id < 1) {
        *result += l_sum[local_id] + l_sum[local_id+1];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

__kernel void q(
        __global const unsigned short* p_c6,
        __global const unsigned* p_c0,
        __global const unsigned* p_c1,
        __global const char* p_c2,
        __global const int* p_c3,
        __global const char* p_c4,
        __global const char* p_c5,
        const unsigned short custom_date,
        unsigned gend,
        unsigned rounds_per_thread,
        __global long8* globalResult) {
    __local long8 l_thread_data[RESULT_SIZE];
    long8 p_thread_data[RESULT_SIZE];

    for(int i = 0;i<RESULT_SIZE;++i) {
        p_thread_data[i] = (long8)(0);
    }
    for(int i = get_local_id(0);i<RESULT_SIZE;i+=get_local_size(0)) {
        l_thread_data[i] = (long8)(0);
    }

    unsigned lstart = (get_group_id(0)*get_local_size(0)*rounds_per_thread)+get_local_id(0);
    unsigned lend = lstart + get_local_size(0)*rounds_per_thread;
    if(lend>gend) lend=gend;

    for(unsigned i0=lstart;i0<lend;i0+=get_local_size(0))
    {
        unsigned short c6=p_c6[i0];
        if (!(c6<=custom_date))
            continue;
            //rkey=RESULT_SIZE+1;

        unsigned c0=p_c0[i0];
        unsigned c1=p_c1[i0];
        unsigned rkey=c0+3*c1;

        char c2=p_c2[i0];
        int c3=p_c3[i0];
        char c4=p_c4[i0];
        char c5=p_c5[i0];

        p_thread_data[rkey]+=(long8)(c2,c3,c3*(100-c4),((long)c3*(100-c4))*(100+c5),c4,1,0,0);
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
        __global long8* interimResult,
        __global long8* globalResult,
        unsigned noOfInterimResults) {
    __local long8 l_thread_data[RESULT_SIZE];
    for(int i = get_local_id(0);i<RESULT_SIZE;i+=get_local_size(0)) {
        l_thread_data[i] = (long8)(0);
    }
    unsigned lim = noOfInterimResults/get_local_size(0);
    lim = get_local_size(0)*(lim+1);

    for(unsigned resNumber = get_local_id(0);resNumber<lim;resNumber+=get_local_size(0)) {
        for(unsigned lineNumber=0;lineNumber<RESULT_SIZE;++lineNumber) {
            long8 a=interimResult[resNumber*RESULT_SIZE+lineNumber]*(resNumber<noOfInterimResults);
            local_sum(&l_thread_data[lineNumber],a);
        }
    }

    for(unsigned short result_pos = get_local_id(0);result_pos<RESULT_SIZE;result_pos+=get_local_size(0)) {
        globalResult[result_pos]=l_thread_data[result_pos];
    }
}

