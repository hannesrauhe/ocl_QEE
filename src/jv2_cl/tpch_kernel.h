//Copyright 2012 <Hannes Rauhe (SAP AG)
#ifndef SRC_TPCH_KERNEL_H_
#define SRC_TPCH_KERNEL_H_


#include "tpch_data_load.h"


#include "jv2_cl/QueryKernel.h"
#include "timer.hpp"

class CTpchMetaInfos;

class tpch_kernel : public QueryKernel {
  protected:
    CTpchMetaInfos* metamanager;
    CTpchSimpleData* datamanager;
    bool verify_result;
    bool print_result;

  public:
    tpch_kernel(
            CTpchMetaInfos* mm,
            unsigned dev_type=CL_DEVICE_TYPE_ALL)
        : QueryKernel(dev_type),
          metamanager(mm),
          datamanager(CTpchSimpleData::BuildStaticTpchSimpleData("")),
          verify_result(true),
          print_result(false) {}

    void set_verify_result(bool t) {
        verify_result = t;
    }

    void set_print_result(bool t) {
        print_result = t;
    }

    void run_query(int q, int number_work_groups=140) {
        switch (q) {
          case 1:
              q1(number_work_groups);
              break;
          case 3:
              q3(number_work_groups);
              break;
          case 4:
              q4(number_work_groups);
              break;
          case 5:
              q5(number_work_groups);
              break;
          case 6:
              q6(number_work_groups);
              break;
          case 7:
              q7(number_work_groups);
              break;
          case 12:
              q12(number_work_groups);
              break;
          case 14:
              q14(number_work_groups);
              break;
          default:
              throw("Query not available");
              break;
        }
    }

    void q1(int number_of_workgroups=140, unsigned short custom_date=10471) {
        const int result_size = 6;
        assert(number_of_workgroups>0);

        reset();

        init_kernels("cl/q1_vec.cl",result_size);

        CTimer timer3("Moving data to GPU", false);
        load_data<unsigned short>("lineitem", "l_shipdate");
        load_data<unsigned>("lineitem", "l_returnflag");
        load_data<unsigned>("lineitem", "l_linestatus");
        load_data<char>("lineitem", "l_quantity");
        load_data<int>("lineitem", "l_extendedprice");
        load_data<char>("lineitem", "l_discount");
        load_data<char>("lineitem", "l_tax");
        synchronize();
        timer3.Time();

        cl_long8* res = reinterpret_cast<cl_long8*>(malloc(result_size * sizeof(cl_long8)));

        kernel.setArg(arg_counter++, custom_date);

        exec_kernel(res, result_size, number_of_workgroups);

/*        if(verify_result) {
            const cl_long8 rres[result_size] = { {37734107,5658655440073,537582571348700,55909065222827692,7390291,1478493,0,0},
                 {991417,148750471038,14130821680541,1469649223194375,194633,38854,0,0},
                {37719753,5656804138090,537412926846040,55889619119831932,7395741,1478870,0,0},
                {0,0,0,0,0,0,0,0},
                {74476040,11170172969774,1061182303076056,110367043872497010,14600873,2920374,0,0},
                {0,0,0,0,0,0,0,0}
                    };
            for (int i = 0; i < result_size; ++i) {
                for(int j=0;j<8;++j) {
                    if(res[i].s[j]!=rres[i].s[j]) {
                        char* msg = new char[200]; //TODO memory leak!
                        sprintf(msg,"Wrong result (%d,%d):%ld != %ld",i,j,res[i].s[j],rres[i].s[j]);
                        free(res);
                        throw msg;
                    }
                }
            }
        }*/
        if(print_result) {
            for (int i = 0; i < result_size; ++i) {
                printf("%ld %ld %ld %ld %ld %ld %ld %ld\n",
                        res[i].s[0],
                        res[i].s[1],
                        res[i].s[2],
                        res[i].s[3],
                        res[i].s[4],
                        res[i].s[5],
                        res[i].s[6],
                        res[i].s[7]
);
            }
        }
        free(res);
    }

    void q3(int number_of_workgroups=140) {
        assert(number_of_workgroups>0);
        const int result_size = 10;

        reset();

        CTimer timer1("Compilation", false);
        init_kernels("cl/q3.cl",result_size);
        timer1.Time();

        CTimer timer3("Loading data (from disk) to GPU", false);
        load_data<unsigned short>("orders", "o_orderdate");
        load_index<unsigned>("orders","o_custkey","customer","c_custkey");
        load_data<unsigned>("customer", "c_mktsegment");
        load_data<int>("orders", "o_orderkey");
        //load_data<int>("orders", "o_shippriority"); //not really needed
        load_index<unsigned>("orders", "o_orderkey", "lineitem", "l_orderkey");
        load_data<unsigned short>("lineitem", "l_shipdate");
        load_data<int>("lineitem", "l_extendedprice");
        load_data<char>("lineitem", "l_discount");

        synchronize();
        timer3.Time();

        restype_q3* res = reinterpret_cast<restype_q3*>(malloc(result_size * sizeof(restype_q3)));

        exec_kernel(res, result_size, number_of_workgroups);

        if(verify_result) {
            //todo
        }
        if(print_result) {
            for (int i = 0; i < result_size; ++i) {
                printf("%ld %d\n",
                        res[i].v0, res[i].k0);
            }
        }
        free(res);
    }


    void q4(int number_of_workgroups=140) {
        assert(number_of_workgroups>0);
        const int result_size = 5;

        reset();

        CTimer timer1("Compilation", false);
        init_kernels("cl/q4.cl",result_size);
        timer1.Time();

        CTimer timer3("Loading data (from disk) to GPU", false);
        load_data<unsigned short>("orders", "o_orderdate");
        load_data<unsigned>("orders", "o_orderpriority");
        load_index<unsigned>("orders", "o_orderkey", "lineitem", "l_orderkey");
        load_data<unsigned short>("lineitem", "l_commitdate");
        load_data<unsigned short>("lineitem", "l_receiptdate");
        synchronize();
        timer3.Time();

        cl_long* res = reinterpret_cast<cl_long*>(malloc(result_size * sizeof(cl_long)));

        exec_kernel(res, result_size, number_of_workgroups);

        if(verify_result) {
            const cl_long rres[result_size] = { 10594,10476,10410,10556,10487 };
            for (int i = 0; i < result_size; ++i) {
                if(res[i]!=rres[i]) {
                    char* msg = new char[200]; //TODO memory leak!
                    sprintf(msg,"Wrong result (%d):%ld != %ld",i,res[i],rres[i]);
                    free(res);
                    throw msg;
                }
            }
        }
        if(print_result) {
            for (int i = 0; i < result_size; ++i) {
                printf("%ld\n",
                        res[i]);
            }
        }
        free(res);
    }

    void q5(int number_of_workgroups=140) {
        assert(number_of_workgroups>0);
        const unsigned result_size = 25;

        reset();

        init_kernels("cl/q5.cl",result_size);

        CTimer timer3("Loading data (from disk) to GPU", false);
        load_index<unsigned>("customer","c_nationkey","nation","n_nationkey");
        load_index<unsigned>("nation","n_regionkey","region","r_regionkey");
        load_data<unsigned>("region","r_name");
        load_data<unsigned>("nation","n_name");
        load_data<char>("customer","c_nationkey");
        load_index<unsigned>("customer","c_custkey","orders","o_custkey");
        load_data<unsigned short>("orders","o_orderdate");
        load_index<unsigned>("orders","o_orderkey","lineitem","l_orderkey");
        load_index<unsigned>("lineitem","l_suppkey","supplier","s_suppkey");
        load_data<char>("supplier","s_nationkey");
        load_data<int>("lineitem","l_extendedprice");
        load_data<char>("lineitem","l_discount");
        synchronize();
        timer3.Time();

        cl_long* res = reinterpret_cast<cl_long*>(malloc(result_size * sizeof(cl_long)));

        exec_kernel(res, result_size, number_of_workgroups);

        if(verify_result) {
            //TODO
        }
        if(print_result) {
            for (unsigned i = 0; i < result_size; ++i) {
                if(res[i]) {
                    printf("%ld\n",
                        res[i]);
                }
            }
        }
        free(res);
    }

    void q6(int number_of_workgroups=140) {
        assert(number_of_workgroups>0);
        unsigned result_size = 1;

        reset();

        init_kernels("cl/q6.cl",result_size);

        CTimer timer2("Loading data from disk", false);
        load_data<unsigned short>("lineitem", "l_shipdate");
        load_data<char>("lineitem", "l_discount");
        load_data<char>("lineitem", "l_quantity");
        load_data<int>("lineitem", "l_extendedprice");
        synchronize();
        timer2.Time();

        cl_long* res = reinterpret_cast<cl_long*>(malloc(result_size * sizeof(cl_long)));

        exec_kernel(res, result_size, number_of_workgroups);

        if(verify_result) {
            const cl_long rres = 1231410782283;
            if(res[0]!=rres) {
                char* msg = new char[200]; //TODO memory leak!
                sprintf(msg,"Wrong result (%d):%ld != %ld",0,res[0],rres);
                free(res);
                throw msg;
            }
        }
        if(print_result) {
            for (unsigned i = 0; i < result_size; ++i) {
                if(res[i]) {
                    printf("%ld\n",
                            res[0]);
                }
            }
        }
        free(res);
    }

    void q7(int number_of_workgroups=140) {
        assert(number_of_workgroups>0);
        unsigned result_size = 200;

        reset();

        init_kernels("cl/q7.cl",result_size);

        CTimer timer2("Loading data from disk", false);
        load_index<unsigned>("supplier","s_nationkey","nation","n_nationkey");
        load_data<unsigned>("nation", "n_name");
        load_index<unsigned>("supplier","s_suppkey","lineitem","l_suppkey");
        load_data<unsigned short>("lineitem", "l_shipdate");
        load_index<unsigned>("lineitem","l_orderkey","orders","o_orderkey");
        load_index<unsigned>("orders","o_custkey","customer","c_custkey");
        load_index<unsigned>("customer","c_nationkey","nation","n_nationkey");
        load_data<unsigned>("nation", "n_name");
        load_data<int>("lineitem", "l_extendedprice");
        load_data<char>("lineitem", "l_discount");
        synchronize();
        timer2.Time();

        cl_long* res = reinterpret_cast<cl_long*>(malloc(result_size * sizeof(cl_long)));

        exec_kernel(res, result_size, number_of_workgroups);

        if(verify_result) {
            const cl_long rres = 1092728160412;
            if(res[0]!=rres) {
                char* msg = new char[200]; //TODO memory leak!
                sprintf(msg,"Wrong result (%d):%ld != %ld",0,res[0],rres);
                free(res);
                throw msg;
            }
        }
        if(print_result) {
            for (unsigned i = 0; i < result_size; ++i) {
                if(res[i]) {
                    printf("%ld\n",
                                    res[i]);
                }
            }
        }
        free(res);
    }

    void q12(int number_of_workgroups=140) {
        assert(number_of_workgroups>0);
        const int result_size = 10;

        reset();

        init_kernels("cl/q12_vec.cl",result_size);

        CTimer timer2("Loading data from disk", false);
        load_data<unsigned>("orders","o_orderpriority");
        load_index<unsigned>("orders","o_orderkey","lineitem","l_orderkey");
        load_data<unsigned>("lineitem","l_shipmode");
        load_data<unsigned short>("lineitem", "l_receiptdate");
        load_data<unsigned short>("lineitem", "l_commitdate");
        load_data<unsigned short>("lineitem", "l_shipdate");
        synchronize();
        timer2.Time();

        cl_long2* res = reinterpret_cast<cl_long2*>(malloc(result_size * sizeof(cl_long2)));

        exec_kernel(res, result_size, number_of_workgroups);

        if(verify_result) {
            const cl_long2 rres[result_size] = { {6202,9324},
             {6200,9262}
            };
            for (int i = 0; i < result_size; ++i) {
                for(int j=0;j<2;++j) {
                    if(res[i].s[j]!=rres[i].s[j]) {
                        char* msg = new char[200]; //TODO memory leak!
                        sprintf(msg,"Wrong result (%d,%d):%ld != %ld",i,j,res[i].s[j],rres[i].s[j]);
                        free(res);
                        throw msg;
                    }
                }
            }
        }
        if(print_result) {
            for (int i = 0; i < result_size; ++i) {
                printf("%ld %ld\n",
                                    res[i].s[0],
                                    res[i].s[1]
                                             );
            }
        }
        free(res);
    }

    void q14(int number_of_workgroups=140) {
        assert(number_of_workgroups>0);
        const int result_size = 1;

        reset();

        init_kernels("cl/q14_vec.cl",result_size);

        CTimer timer2("Loading data from disk", false);
        load_data<unsigned short>("lineitem","l_shipdate");
        load_index<unsigned>("lineitem","l_partkey","part","p_partkey");
        load_data<int>("lineitem","l_extendedprice");
        load_data<char>("lineitem", "l_discount");
        load_data<unsigned>("part", "p_type");
        synchronize();
        timer2.Time();

        cl_long2* res = reinterpret_cast<cl_long2*>(malloc(result_size * sizeof(cl_long2)));

        exec_kernel(res, result_size, number_of_workgroups);


        if(verify_result) {
            const cl_long2 rres[result_size] = { {4524288052301, 27619493282271}        };
            for (int i = 0; i < result_size; ++i) {
                for(int j=0;j<2;++j) {
                    if(res[i].s[j]!=rres[i].s[j]) {
                        char* msg = new char[200]; //TODO memory leak!
                        sprintf(msg,"Wrong result (%d,%d):%ld != %ld",i,j,res[i].s[j],rres[i].s[j]);
                        free(res);
                        throw msg;
                    }
                }
            }
        }
        if(print_result) {
            for (int i = 0; i < result_size; ++i) {
                printf("%ld %ld\n",
                                    res[i].s[0],
                                    res[i].s[1]
                                             );
            }
        }
        free(res);
    }

    void transfertest() {
        reset();
    	//loading fake kernel
    	init_kernels("cl/q14_vec.cl",1);
    	load_data<int>("lineitem","l_extendedprice");
    	printf("col_size: %d * %ld byte = %ld bytes\n",data_size,sizeof(int),data_size*sizeof(int));
    }

  protected:
    template <class T>
    void load_data(
            const char* table,
            const char* column) {
      int c                              = datamanager->GetColumnID(datamanager->GetGlobalTableID(
      table), column);
      ColData coldata = datamanager->GetColumnData(c);
      assert(coldata.alen);
      if(!data_size) {
      data_size = coldata.alen;
    }
      ptr_as_kernel_arg(coldata.aptr, (coldata.alen)*sizeof(T),c);
    }

    template <class T>
    void load_index(
            const char* table,
            const char* column,
            const char* table2,
            const char* column2) {
        int c1 = datamanager->GetColumnID(datamanager->GetGlobalTableID(table), column);
        int c2 = datamanager->GetColumnID(datamanager->GetGlobalTableID(table2), column2);
        IndexData res = datamanager->GetIndexData(c1,c2);
        assert(res.alen);
        if(!data_size) {
      data_size = res.alen;
        }
        ptr_as_kernel_arg(res.aptr, (res.alen)*sizeof(T),c1,res.optr,(res.olen)*sizeof(T),c2);
    }
};

#endif  // SRC_TPCH_KERNEL_H_
