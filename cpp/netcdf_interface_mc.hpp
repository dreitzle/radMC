/*
 * netcdf_interface.hpp
 *
 *  Created on: Aug 4, 2016
 *      Author: smueller
 */

#ifndef UTILS_NETCDF_INTERFACE_MC_HPP_
#define UTILS_NETCDF_INTERFACE_MC_HPP_

#include <string>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <netcdf.h>
#include <stdlib.h>

class netcdf_interface
{
private:

    int nc_id = 0;
    int deflate_level;
    std::string filename;

public:

    /* constructor */
    netcdf_interface(const std::string & filename_, int deflate_level_ = 0);

    void create_file(int cmode = NC_CLOBBER | NC_NETCDF4);
    void create_file(const std::string & filename_, int cmode = NC_CLOBBER | NC_NETCDF4);
    void put_attr(const std::string & name, float attribute);
    void put_attr(const std::string & name, double attribute);
    void put_attr(const std::string & name, int attribute);
    void put_attr(const std::string & name, unsigned int attribute);
    void put_attr(const std::string & name, long long attribute);
    void put_attr(const std::string & name, unsigned long long attribute);
    void put_attr(const std::string & name, const std::string & attribute);
    void put_attr(const std::string & name, nc_type xtype, const void * value, size_t len = 1);
    void put_attr(const std::string & name, const std::vector<float> & attribute, nc_type type = NC_FLOAT);
    void put_attr(const std::string & name, const std::vector<double> & attribute, nc_type type = NC_DOUBLE);
    void put_attr(const std::string & name, const std::vector<int> & attribute, nc_type type = NC_INT);
    void put_attr(const std::string & name, const std::vector<unsigned> & attribute, nc_type type = NC_UINT);
    void put_attr(const std::string & name, const std::vector<long long> & attribute, nc_type type = NC_INT64);
    void put_attr(const std::string & name, const std::vector<unsigned long long> & attribute, nc_type type = NC_UINT64);
    void put_attr(const std::string & name, const std::vector<std::string> & attribute);

    std::vector<int> def_dimensions(const std::vector<std::string> & dimensionnames, const std::vector<size_t> & dimensionsizes);
    int def_variable(const std::string & name, nc_type xtype ,const std::vector<int> & dimids);
    int def_variable(const std::string & name, nc_type xtype ,int dimid);

    void enddef();

    int put_variable(const std::string & name, const void * data);
    int put_variable(int varid, const void * data);
    int put_variable_auto(int varid, const float * data);
    int put_variable_auto(int varid, const double * data);
    int put_variable_auto(int varid, const int * data);
    int put_variable_auto(int varid, const long long * data);
    int put_variable_auto(int varid, const unsigned * data);
    int put_variable_auto(int varid, const unsigned long long * data);
    template<typename T>
    int put_variable_auto(const std::string & name, const T * data);

    int put_vara(const std::string & name, const std::vector<size_t> & startp, const std::vector<size_t> & countp, const void * data);

    void close_file();

    void open_file(int cmode = NC_NOWRITE);
    void open_file(const std::string & filename_, int cmode = NC_NOWRITE);
    void get_attr(const std::string & name, float* attribute);
    void get_attr(const std::string & name, double* attribute);
    void get_attr(const std::string & name, int* attribute);
    void get_attr(const std::string & name, unsigned int* attribute);
    void get_attr(const std::string & name, long long* attribute);
    void get_attr(const std::string & name, unsigned long long* attribute);
    void get_attr(const std::string & name, void * value);
    void get_attr(const std::string & name, std::string & attribute);
    size_t get_attlen(const std::string & name);
    void get_attr(const std::string & name, std::vector<float> & attribute);
    void get_attr(const std::string & name, std::vector<double> & attribute);
    void get_attr(const std::string & name, std::vector<int> & attribute);
    void get_attr(const std::string & name, std::vector<unsigned> & attribute);
    void get_attr(const std::string & name, std::vector<long long> & attribute);
    void get_attr(const std::string & name, std::vector<unsigned long long> & attribute);
    void get_attr(const std::string & name, std::vector<std::string> & attribute);
    nc_type get_attr_type(const std::string & name);

    int get_variable(const std::string & name, void * data);
    int get_variable(int varid, void * data);
    int get_variable_auto(int varid, float * data);
    int get_variable_auto(int varid, double * data);
    int get_variable_auto(int varid, int * data);
    int get_variable_auto(int varid, long long * data);
    int get_variable_auto(int varid, unsigned * data);
    int get_variable_auto(int varid, unsigned long long * data);
    template<typename T>
    int get_variable_auto(const std::string & name, T * data);
    int get_variable_string(int varid, char ** data);

    template<typename T>
    int get_variable_vector(const std::string & name, std::vector<T> & variable);

    int get_vara(const std::string & name, const std::vector<size_t> & startp, const std::vector<size_t> & countp, void * data);

    nc_type get_variable_type(const std::string & name);
    nc_type get_variable_type(int varid);

    size_t get_variable_size(const std::string & name);
    size_t get_variable_size(int varid);
    std::vector<size_t> get_variable_sizes(const std::string & name);
    std::vector<size_t> get_variable_sizes(int varid);

    std::vector<std::string> get_variable_dimnames(int varid);

    int get_variable_id(const std::string & name);
    int get_id();

    bool variable_defined(const std::string & name);

    inline void checkErr(const int err, const char * name) const;

};

inline void netcdf_interface::checkErr(const int err, const char * name) const
{
    if (err != NC_NOERR) {
        std::cerr << "ERROR: " << name  << " (" << nc_strerror(err) << ")" << std::endl;
        nc_close(nc_id);
        exit(EXIT_FAILURE);
    }
}


#endif /* UTILS_NETCDF_INTERFACE_MC_HPP_ */
