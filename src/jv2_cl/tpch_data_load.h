//Copyright 2013 <Jonathan Dees (SAP AG)
#ifndef SRC_TPCH_DATA_LOAD_H_
#define SRC_TPCH_DATA_LOAD_H_

#include <string>
#include <cstdint>

struct IndexData
{
    uint64_t alen;
    void* aptr;
    uint64_t olen;
    void* optr;
};
struct ColData
{
    uint64_t alen;
    void* aptr;
};

static_assert( sizeof(void*)==8, "we assume pointer size of 8 bytes for deserialization" );

class CTpchSimpleData
{
  private:
    std::string path;
    CTpchSimpleData(const std::string& path) : path(path) {}
    void init();
  public:
    static CTpchSimpleData* BuildStaticTpchSimpleData(const std::string& path)
    {
      // owernship to caller
      return new CTpchSimpleData(path);
    }
    int GetGlobalTableID(std::string);
    int GetColumnID(int,std::string);
    IndexData GetIndexData(int,int);
    ColData GetColumnData(int);
};


#endif

