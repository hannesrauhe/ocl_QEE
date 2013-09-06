//Copyright 2013 <Jonathan Dees (SAP AG)
#include "tpch_data_load.h"

int CTpchSimpleData::GetGlobalTableID(std::string s)
{
  int32_t res;
  FILE* f;
  f = fopen( (s+".tableid.bin").c_str(), "rb" );
  fread(&res,sizeof(int32_t),1,f);
  fclose(f);
  return res;
}

int CTpchSimpleData::GetColumnID(int id,std::string s)
{
  int32_t res;
  FILE* f;
  f = fopen( (std::to_string(id)+"_"+s+".colid.bin").c_str(), "rb" );
  fread(&res,sizeof(int32_t),1,f);
  fclose(f);
  return res;
}


IndexData CTpchSimpleData::GetIndexData(int id1,int id2)
{
  IndexData res;
  FILE* f;
  f = fopen( (std::to_string(id1)+"_"+std::to_string(id2)+".indexdata.bin").c_str(), "rb" );

  int64_t valsize1;
  fread(&valsize1,sizeof(int64_t),1,f);
  fread(&res.alen,sizeof(int64_t),1,f);
  res.aptr=malloc(valsize1*res.alen);
  fread(res.aptr,valsize1,res.alen,f);

  int64_t valsize2;
  fread(&valsize2,sizeof(int64_t),1,f);
  fread(&res.olen,sizeof(int64_t),1,f);
  res.optr=malloc(valsize2*res.olen);
  fread(res.optr,valsize2,res.olen,f);
  fclose(f);
  return res;
}


ColData CTpchSimpleData::GetColumnData(int id)
{
  ColData res;
  FILE* f;
  f = fopen( (std::to_string(id)+".coldata.bin").c_str(), "rb" );
  int64_t valsize1;
  fread(&valsize1,sizeof(int64_t),1,f);
  fread(&res.alen,sizeof(int64_t),1,f);
  res.aptr=malloc(valsize1*res.alen);
  fread(res.aptr,valsize1,res.alen,f);
  fclose(f);
  return res;
}
