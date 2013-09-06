//Copyright Hannes Rauhe (SAP AG)

#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include <memory>

#include "jv2_cl/tpch_kernel.h"

#ifndef SERVER_PORT
#define SERVER_PORT 22501
#endif

static const unsigned device_type = CL_DEVICE_TYPE_CPU;
static const int default_number_threads = 256;
static const bool clear_cache_after_exec = true;
static const bool verify_res = false;
static const bool print_res = true;
static const bool use_opt = true;
char path[255] = "bin/";

void error(const char* msg) {
    perror(msg);
    exit(1);
}

void init_tpch_k(tpch_kernel& t) {
    t.set_num_threads(default_number_threads);
    t.set_opt(use_opt);
    t.set_print_result(print_res);
    t.set_verify_result(verify_res);
}

void one_time_only_mode(
        int argc,
        char** argv) {
    if (argc < 3) {
      printf("Usage: %s <querynumber> <work_factor>\n" // [<Path_to_db>]\n"
             , argv[0]);
        exit(1);
    }

    if(strcmp(argv[1],"transfertest")==0) {
      CTpchMetaInfos* metamanager = nullptr;
    	tpch_kernel t(metamanager, device_type);
    	t.transfertest();
    	t.print_exec_times();
    	exit(0);
    }

    int  query       = atoi(argv[1]);
    int  number_of_workgroups = atoi(argv[2]);
    if (argc > 3)
        strcpy(path, argv[3]);

    CTpchMetaInfos* metamanager = nullptr;

    tpch_kernel t(metamanager, device_type);
    init_tpch_k(t);
    t.run_query(query,number_of_workgroups);
    t.print_exec_times();
    printf("SUCCESS;%f;%f;%f\n",
            t.get_exec_time_by_category("transfer"),t.get_exec_time_by_category("kernelExec"),t.get_exec_time_by_category("kernelReduce"));
}

void server_mode(
        int argc,
        char* argv[]) {
    int                sockfd, newsockfd=-1;
    socklen_t          clilen;
    char               buffer[256];
    struct sockaddr_in serv_addr, cli_addr;
    int                n;
    bool               run       = true;
    if (argc > 1)
        strcpy(path, argv[1]);

    CTpchMetaInfos* metamanager = nullptr;
    std::unique_ptr<tpch_kernel> t(new tpch_kernel(metamanager, device_type));
    init_tpch_k(*t);

    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    int optval = 1;
    setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof optval);
    if (sockfd < 0)
        error("ERROR opening socket");
    bzero((char*) &serv_addr, sizeof(serv_addr));
    serv_addr.sin_family      = AF_INET;
    serv_addr.sin_addr.s_addr = INADDR_ANY;
    serv_addr.sin_port        = htons(SERVER_PORT);
    if (bind(sockfd, (struct sockaddr*) &serv_addr,
                sizeof(serv_addr)) < 0)
        error("ERROR on binding");
    while (run) {
        try {
            int query = 0;
            int number_work_groups = 0;
            double total_seconds = 0;
            char resultmsg[255];

            if(listen(sockfd, 5)==-1);
            clilen    = sizeof(cli_addr);
            newsockfd = accept(sockfd,
                        (struct sockaddr*) &cli_addr,
                        &clilen);
            if (newsockfd < 0) {
                throw("ERROR on accept");
            }
            bzero(buffer, 256);
            n = read(newsockfd, buffer, 255);
            if (n < 0) {
                throw("ERROR reading from socket");
            }
            printf("\n----------------------\nGot Message:%s\n",buffer);

            if(strstr(buffer,"exit")) {
                run = false;
                sprintf(resultmsg,"Bye");
            } else {
                char* pch = strtok(buffer, " ,.-");
                if (pch == NULL) {
                    throw("Message was wrong");
                }
                query = atoi(pch);
                pch = strtok(NULL, " ,.-");
                if (pch == NULL) {
                    printf("Message was wrong");
                    close(newsockfd);
                    continue;
                }
                number_work_groups = atoi(pch);
                CTimer total("total",false);

                t->run_query(query,number_work_groups);

                total_seconds = total.GetTimeS();
                t->print_exec_times();
                sprintf(resultmsg,"SUCCESS;%d;%d;%f;%f;%f;%f",
                        query,number_work_groups,
                        t->get_exec_time_by_category("transfer"),t->get_exec_time_by_category("kernel"),t->get_exec_time_by_category("kernelReduce"),total_seconds);
            }
            printf("SEND: %s\n",resultmsg);
            n = write(newsockfd,resultmsg,strlen(resultmsg));
            if (n < 0) error("ERROR writing to socket");
        } catch(char const* msg) {
            printf("%s\n",msg);
            if(newsockfd>0) {
                n = write(newsockfd,msg,strlen(msg));
                if (n < 0) error("ERROR writing to socket");
            }
            printf("Trying to reinit OpenCL-Platform\n");
            t.reset(new tpch_kernel(metamanager, device_type));
            init_tpch_k(*t);
        } catch(...) {
            const char* msg = "Unknown Error occured";
            printf("%s\n",msg);
            if(newsockfd>0) {
                n = write(newsockfd,msg,strlen(msg));
                if (n < 0) error("ERROR writing to socket");
            }
            throw;
        }
        if(clear_cache_after_exec) {
            printf("Freeing cache on GPU\n");
            t->clear_data_from_device();
        }
        close(newsockfd);
    }
    close(sockfd);
}

int main(
        int argc,
        char* argv[]) {
    printf("Usage for direct execution: %s <querynumber> <work_factor> [<Path_to_db>]\n", argv[0]);
    printf("Usage for server listening mode: %s [<Path_to_db>]\n", argv[0]);
    printf("NOTICE: Make sure, cl-files are present in ./cl\n");
    try {
        if (argc >= 3)
            one_time_only_mode(argc, argv);
        else
            server_mode(argc, argv);
    } catch(const cl::Error& er) {
        printf("CL ERROR (should have been caught earlier): %s(%d)\n", er.what(), er.err());
    } catch(const char* err) {
        error(err);
    }

    printf("Done\n");
    return 0;
}
