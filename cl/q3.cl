#define JV2_MAX_VALUE -9223372036854775806

struct restype_q3 {
    long v0;
    unsigned k0;
};

inline void ComparatorLocal(
    __local struct restype_q3 *keyA,
    __local struct restype_q3 *keyB,
    uint arrowDir
){
    if( (keyA->v0 > keyB->v0) == arrowDir ){
        struct restype_q3 t;
        t = *keyA; *keyA = *keyB; *keyB = t;
    }
}

#define LOCAL_SIZE_LIMIT NUM_OF_THREADS * 2
__kernel __attribute__((reqd_work_group_size(LOCAL_SIZE_LIMIT / 2, 1, 1)))
void local_sort(
        __local struct restype_q3* l_key
){
    uint arrayLength = LOCAL_SIZE_LIMIT;
    uint sortDir = 0;

    for(uint size = 2; size < arrayLength; size <<= 1){
        //Bitonic merge
        uint dir = ( (get_local_id(0) & (size / 2)) != 0 );
        for(uint stride = size / 2; stride > 0; stride >>= 1){
            barrier(CLK_LOCAL_MEM_FENCE);
            uint pos = 2 * get_local_id(0) - (get_local_id(0) & (stride - 1));
            ComparatorLocal(
                &l_key[pos +      0],
                &l_key[pos + stride],
                dir
            );
        }
    }

    //dir == sortDir for the last bitonic merge step
    {
        for(uint stride = arrayLength / 2; stride > 0; stride >>= 1){
            barrier(CLK_LOCAL_MEM_FENCE);
            uint pos = 2 * get_local_id(0) - (get_local_id(0) & (stride - 1));
            ComparatorLocal(
                &l_key[pos +      0],
                &l_key[pos + stride],
                sortDir
            );
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
}

__kernel void q(
        __global const unsigned short* p_c1,       //o_orderdate
        __global const unsigned*        t0_to_t1,    //orders2customer
        __global const unsigned*        p_c5,       //c_mktsegment
        __global const int*             p_c0,        //o_orderkey
        //__global const int*             p_c2,        //o_shippriority
        __global const unsigned*        mo_t0_to_t2, //orders 2 lineitem
        __global const unsigned short*  p_c6,        //l_shipdate
        __global const int*             p_c3,        //l_extendenprice
        __global const char*            p_c4,        //l_discount
        unsigned gend,
        unsigned rounds_per_thread,
        __global struct restype_q3* globalResult) {
    __local struct restype_q3 l_thread_data[LOCAL_SIZE_LIMIT];
    struct restype_q3 p_thread_data[RESULT_SIZE];

    for(int i = 0;i<RESULT_SIZE;++i) {
        p_thread_data[i].v0 = JV2_MAX_VALUE;;
    }
//    for(int i = get_local_id(0);i<NUM_OF_THREADS;i+=get_local_size(0)) {
//        l_thread_data[i] = (long)(0);
//    }

    unsigned lstart = (get_group_id(0)*get_local_size(0)*rounds_per_thread)+get_local_id(0);
    unsigned lend = lstart + get_local_size(0)*rounds_per_thread;
    if(lend>gend) lend=gend;

    short res_number=0;
    short smallest_res=0;

    for(unsigned i0=lstart;i0<lend;i0+=get_local_size(0))
    {
        unsigned short c1 = p_c1[i0];
        if (!(c1 < 9204))
            continue;
        unsigned i1 = t0_to_t1[i0];
        unsigned c5 = p_c5[i1];
        if (!(c5 == 1))
            continue;
        int c0 = p_c0[i0];
        unsigned mo2rend = mo_t0_to_t2[i0+1];
        long agg0 = 0;
        for (unsigned i2 = mo_t0_to_t2[i0]; i2 < mo2rend; ++i2) {
            unsigned short c6 = p_c6[i2];
            if (!(c6 > 9204))
                continue;
            int c3 = p_c3[i2];
            char c4 = p_c4[i2];
            agg0 += (c3*(100-c4));
        }
        if(res_number<RESULT_SIZE) {
            p_thread_data[res_number].v0=agg0;
            p_thread_data[res_number].k0=i0;
            if(agg0<p_thread_data[smallest_res].v0) {
                smallest_res=res_number;
            }
            ++res_number;
        } else {
            if(agg0>p_thread_data[smallest_res].v0) {
                p_thread_data[smallest_res].v0=agg0;
                p_thread_data[smallest_res].k0=i0;
                for(short i=0;i<RESULT_SIZE;++i) {
                    if(p_thread_data[i].v0<p_thread_data[smallest_res].v0) {
                        smallest_res=i;
                    }
                }
            }
        }
    }

    l_thread_data[get_local_id(0)]=p_thread_data[0];
    for(short j=1;j<RESULT_SIZE;j++) {
        l_thread_data[get_local_id(0)+NUM_OF_THREADS]=p_thread_data[j];

        barrier(CLK_LOCAL_MEM_FENCE);
        local_sort(l_thread_data);
    }

    for(int i = get_local_id(0);i<RESULT_SIZE;i+=get_local_size(0)) {
        unsigned result_pos = i + get_group_id(0)*RESULT_SIZE;
        globalResult[result_pos]=l_thread_data[i];
    }
}

__kernel void q_reduce(
        __global struct restype_q3* interimResult,
        __global struct restype_q3* globalResult,
        unsigned noOfInterimResults) {
    __local struct restype_q3 l_thread_data[LOCAL_SIZE_LIMIT];

    unsigned lim = noOfInterimResults/get_local_size(0);
    lim = get_local_size(0)*(lim+1);

    if(get_local_id(0)<RESULT_SIZE) {
        l_thread_data[get_local_id(0)]=interimResult[get_local_id(0)];
    }

    for(short j=get_local_id(0); j-get_local_id(0)<noOfInterimResults*RESULT_SIZE; j+=LOCAL_SIZE_LIMIT-RESULT_SIZE) {
        if(get_local_id(0)>=RESULT_SIZE) {
            if(j<noOfInterimResults*RESULT_SIZE) {
                l_thread_data[get_local_id(0)]=interimResult[j];
            } else {
                l_thread_data[get_local_id(0)].v0=JV2_MAX_VALUE;
            }
        }
        if((j+NUM_OF_THREADS<noOfInterimResults*RESULT_SIZE)) {
            l_thread_data[NUM_OF_THREADS+get_local_id(0)]=interimResult[NUM_OF_THREADS+j];
        } else {
            l_thread_data[NUM_OF_THREADS+get_local_id(0)].v0=JV2_MAX_VALUE;
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        local_sort(l_thread_data);
    }

    for(unsigned short result_pos = get_local_id(0);result_pos<RESULT_SIZE;result_pos+=get_local_size(0)) {
        globalResult[result_pos]=l_thread_data[result_pos];
    }
}
