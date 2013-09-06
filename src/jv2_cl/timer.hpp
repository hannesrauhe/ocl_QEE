#pragma once
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/date_time/microsec_time_clock.hpp>
#include <sstream>
#include <string>

/// switch on off with #define MEASURE_TIME

#define MEASURE_TIME

class CTimer {
public:
#ifdef MEASURE_TIME
  
  explicit CTimer( const std::string& name, bool verbose = true ) : verbose( verbose ), number( 1 ), name( name ), lasttime( current_microsec_time() )
  {
    if (verbose)
      printf("Time S (%s)\n", this->name.c_str());
  }
  void Time()
  {
      printf("Time %d (%s): ", number++, this->name.c_str() );
      print_time();
      printf("\n");
  }
  void Time( const std::string& label )
  {
      printf("Time %s (%s): ", label.c_str(), this->name.c_str() );
      print_time();
      printf("\n");
  }
  double GetTimeS()
  {
    return ( current_microsec_time() - lasttime ) / 1000000.0;
  }
  double GetTimeSReset()
  {
    signed long long tmp = current_microsec_time();
    double res = ( tmp - lasttime ) / 1000000.0;
    lasttime = tmp;
    return res;
  }
  ~CTimer()
  {
    if (verbose)
    {
      printf("Time E (%s): ", this->name.c_str() );
      print_time();
      printf("\n");
    }
  }
private:
  bool verbose;
  int number;
  const std::string name;
  signed long long lasttime;
  signed long long current_microsec_time() const
  {
   signed long long res = 
     boost::posix_time::time_duration(
         boost::posix_time::microsec_clock::local_time() -
       boost::posix_time::ptime(
 boost::gregorian::date( 1970,boost::date_time::Jan,1) ) 
       ).total_microseconds();
    return res;
  }
  void print_time()
  {
      double diff = ( current_microsec_time() - lasttime ) / 1000000.0;
      printf("%.6lf", diff );
  }
#else
public:
  explicit CTimer( const std::string& /* nothin */, bool verbose = true )
  {
  }
  void Time()
  {
  }
  void Time( const std::string& label )
  {
  }
  double GetTimeS()
  {
    return 0.0;
  }
#endif

};
