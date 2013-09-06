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
        __global unsigned*        t0_to_t4,
        __global unsigned*        t4_to_t5,
        __global unsigned*        p_c5,
        __global unsigned*        p_c0,
        __global char*            p_c3,
        __global unsigned*        mo_t0_to_t1,
        __global unsigned*        mi_t0_to_t1,// join multi index inverted customer_to_orders
        __global unsigned short*  p_c6,
        __global unsigned*        mo_t1_to_t2,// join multi index inverted orders_to_lineitem
        __global unsigned*        t2_to_t3,// join forward lineitem_to_supplier
        __global char*            p_c4,
        __global int*             p_c1,
        __global char*            p_c2,
        unsigned gend,
        unsigned rounds_per_thread,
        __global long* globalResult) {
    __local long l_thread_data[RESULT_SIZE];
    long p_thread_data[RESULT_SIZE];
    for(int i = 0;i<RESULT_SIZE;++i) {
        p_thread_data[i] = (long)(0);
    }

    for(int i = get_local_id(0);i<RESULT_SIZE;i+=get_local_size(0)) {
        l_thread_data[i] = (long)(0);
    }

    unsigned lstart = (get_group_id(0)*get_local_size(0)*rounds_per_thread)+get_local_id(0);
    unsigned lend = lstart + get_local_size(0)*rounds_per_thread;
    if(lend>gend) lend=gend;

    for(unsigned i0=lstart;i0<lend;i0+=get_local_size(0))
    {
        unsigned i4 = t0_to_t4[i0];
        unsigned i5 = t4_to_t5[i4];
        unsigned c5 = p_c5[i5];
        if (!(c5 == 2))
            continue;
        unsigned c0 = p_c0[i4];
        unsigned rkey = i4;
        char     c3   = p_c3[i0];
        unsigned mo1end = mo_t0_to_t1[i0+1];
        for (unsigned mi1 = mo_t0_to_t1[i0]; mi1 < mo1end; ++mi1) {
            unsigned i1 = mi_t0_to_t1[mi1];
            unsigned short c6 = p_c6[i1];
            if (!(c6 >= 8766))
                continue;
            if (!(c6 < 9131))
                continue;
            unsigned mo2rend = mo_t1_to_t2[i1+1];
            for (unsigned i2 = mo_t1_to_t2[i1]; i2 < mo2rend; ++i2) {
                unsigned i3 = t2_to_t3[i2];
                char c4 = p_c4[i3];
                if (!(c3 == c4))
                    continue;
                int c1 = p_c1[i2];
                char c2 = p_c2[i2];
                long agg0 = (c1*(100-c2));
                p_thread_data[rkey] += agg0;
            }
        }
    }

    for(int j=0;j<RESULT_SIZE;++j) {
        local_sum(&l_thread_data[j],p_thread_data[j]);
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
