//TODO correct errors according to q1_vcl.cl

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable

struct ResLine {
    long v0;
    long v1;
    long v2;
    long v3;
    long v4;
    long v5;
};

__kernel void copy_test(
        __global int* dest,
        __global int* src,
        unsigned length) {

    int i = get_global_id(0);

    if(i<length) {
        dest[i]=src[i];
    }

}

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
        __global const unsigned short* p_c6,
        __global const unsigned* p_c0,
        __global const unsigned* p_c1,
        __global const char* p_c2,
        __global const int* p_c3,
        __global const char* p_c4,
        __global const char* p_c5,
        unsigned gend,
        __global struct ResLine* globalResult) {
    __local struct ResLine l_thread_data[RESULT_SIZE];

    for(int i = get_local_id(0);i<RESULT_SIZE;i+=512) {
        l_thread_data[i].v0=0;
        l_thread_data[i].v1=0;
        l_thread_data[i].v2=0;
        l_thread_data[i].v3=0;
        l_thread_data[i].v4=0;
        l_thread_data[i].v5=0;
    }

    int agg5=1;
    unsigned lstart = get_global_id(0);
    unsigned lend = lstart + get_local_size(0);
    if(lend>gend) lend=gend;

    for(unsigned i0=lstart;i0<lend;i0+=get_local_size(0))
    {
        unsigned short c6=p_c6[i0];

        if (!(c6<=10471)) continue;

        unsigned c0=p_c0[i0];

        unsigned c1=p_c1[i0];

        unsigned rkey=c0+3*c1;
        char c2=p_c2[i0];

        int agg0=c2;
        int c3=p_c3[i0];

        int agg1=c3;
        char c4=p_c4[i0];

        int agg4=c4;
        char c5=p_c5[i0];

        int agg3=(((long)c3*(100-c4))*(100+c5));
        int agg2=(c3*(100-c4));

        barrier(CLK_LOCAL_MEM_FENCE);

        for(int j=0;j<RESULT_SIZE;++j) {
            local_sum(&l_thread_data[j].v0,agg0*(rkey==j));
            local_sum(&l_thread_data[j].v1,agg1*(rkey==j));
            local_sum(&l_thread_data[j].v2,agg2*(rkey==j));
            local_sum(&l_thread_data[j].v3,agg3*(rkey==j));
            local_sum(&l_thread_data[j].v4,agg4*(rkey==j));
            local_sum(&l_thread_data[j].v5,agg5*(rkey==j));
        }
    }



    for(int i = get_local_id(0);i<RESULT_SIZE;i+=get_local_size(0)) {
        unsigned result_pos = i + get_group_id(0)*RESULT_SIZE;
        globalResult[result_pos].v0=l_thread_data[i].v0;
        globalResult[result_pos].v1=l_thread_data[i].v1;
        globalResult[result_pos].v2=l_thread_data[i].v2;
        globalResult[result_pos].v3=l_thread_data[i].v3;//get_num_groups(0);//
        globalResult[result_pos].v4=l_thread_data[i].v4;//get_local_size(0);//
        globalResult[result_pos].v5=l_thread_data[i].v5;
    }
}

__kernel void q_reduce(
        __global struct ResLine* interimResult,
        __global struct ResLine* globalResult,
        unsigned noOfInterimResults) {
    __local struct ResLine l_thread_data[RESULT_SIZE];
//    unsigned short resNumber = get_local_id(0) / RESULT_SIZE;
//    unsigned short lineNumber = get_local_id(0) % RESULT_SIZE;
//    if(resNumber<noOfInterimResults) {
//        long a=interimResult[resNumber*RESULT_SIZE+lineNumber].v0;
//        local_sum(&l_thread_data[lineNumber].v0,a);
//        a=interimResult[resNumber*RESULT_SIZE+lineNumber].v1;
//        local_sum(&l_thread_data[lineNumber].v1,a);
//        a=interimResult[resNumber*RESULT_SIZE+lineNumber].v2;
//        local_sum(&l_thread_data[lineNumber].v2,a);
//        a=interimResult[resNumber*RESULT_SIZE+lineNumber].v3;
//        local_sum(&l_thread_data[lineNumber].v3,a);
//        a=interimResult[resNumber*RESULT_SIZE+lineNumber].v4;
//        local_sum(&l_thread_data[lineNumber].v4,a);
//        a=interimResult[resNumber*RESULT_SIZE+lineNumber].v5;
//        local_sum(&l_thread_data[lineNumber].v5,a);
//    }
    unsigned short resNumber = get_local_id(0);
    for(unsigned short lineNumber=0;lineNumber<RESULT_SIZE;++lineNumber) {
        long a=interimResult[resNumber*RESULT_SIZE+lineNumber].v0;
        local_sum(&l_thread_data[lineNumber].v0,a);
        a=interimResult[resNumber*RESULT_SIZE+lineNumber].v1;
        local_sum(&l_thread_data[lineNumber].v1,a);
        a=interimResult[resNumber*RESULT_SIZE+lineNumber].v2;
        local_sum(&l_thread_data[lineNumber].v2,a);
        a=interimResult[resNumber*RESULT_SIZE+lineNumber].v3;
        local_sum(&l_thread_data[lineNumber].v3,a);
        a=interimResult[resNumber*RESULT_SIZE+lineNumber].v4;
        local_sum(&l_thread_data[lineNumber].v4,a);
        a=interimResult[resNumber*RESULT_SIZE+lineNumber].v5;
        local_sum(&l_thread_data[lineNumber].v5,a);
    }

    for(unsigned short result_pos = get_local_id(0);result_pos<RESULT_SIZE;result_pos+=get_local_size(0)) {
        globalResult[result_pos].v0=l_thread_data[result_pos].v0;
        globalResult[result_pos].v1=l_thread_data[result_pos].v1;
        globalResult[result_pos].v2=l_thread_data[result_pos].v2;
        globalResult[result_pos].v3=l_thread_data[result_pos].v3;//get_num_groups(0);//
        globalResult[result_pos].v4=l_thread_data[result_pos].v4;//get_local_sresult_posze(0);//
        globalResult[result_pos].v5=l_thread_data[result_pos].v5;
    }
}

