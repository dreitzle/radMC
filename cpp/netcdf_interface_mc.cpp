/*
 * netcdf_interface.cpp
 *
 *  Created on: Dec 15, 2016
 *      Author: smueller
 */

#include <netcdf_interface_mc.hpp>
#include <utility>

netcdf_interface::netcdf_interface(const std::string & filename_, int deflate_level_): deflate_level(deflate_level_), filename(filename_)
{}


void netcdf_interface::create_file(int cmode)
{
    checkErr( nc_create(filename.c_str(), cmode, &nc_id) , "nc_create" );
}

void netcdf_interface::create_file(const std::string & filename_, int cmode)
{
    filename = filename_;
    checkErr( nc_create(filename.c_str(), cmode, &nc_id) , "nc_create" );
}

void netcdf_interface::put_attr(const std::string & name, float attribute)
{
    checkErr( nc_put_att_float(nc_id, NC_GLOBAL, name.c_str(), NC_FLOAT, 1, &attribute), "nc_put_att_float" );
}

void netcdf_interface::put_attr(const std::string & name, double attribute)
{
    checkErr( nc_put_att_double(nc_id, NC_GLOBAL, name.c_str(), NC_DOUBLE, 1, &attribute), "nc_put_att_double" );
}

void netcdf_interface::put_attr(const std::string & name, int attribute)
{
    checkErr( nc_put_att_int(nc_id, NC_GLOBAL, name.c_str(), NC_INT, 1, &attribute), "nc_put_att_uint" );
}

void netcdf_interface::put_attr(const std::string & name, unsigned int attribute)
{
    checkErr( nc_put_att_uint(nc_id, NC_GLOBAL, name.c_str(), NC_UINT, 1, &attribute), "nc_put_att_uint" );
}

void netcdf_interface::put_attr(const std::string & name, long long attribute)
{
    checkErr( nc_put_att_longlong(nc_id, NC_GLOBAL, name.c_str(), NC_INT64, 1, &attribute), "nc_put_att_ulonglong" );
}

void netcdf_interface::put_attr(const std::string & name, unsigned long long attribute)
{
    checkErr( nc_put_att_ulonglong(nc_id, NC_GLOBAL, name.c_str(), NC_UINT64, 1, &attribute), "nc_put_att_ulonglong" );
}

void netcdf_interface::put_attr(const std::string & name, const std::string & attribute)
{
    const char * cstring = attribute.c_str();
    checkErr( nc_put_att_string(nc_id, NC_GLOBAL, name.c_str(), 1, &cstring), "nc_put_att_ulonglong" );
}

void netcdf_interface::put_attr(const std::string & name, nc_type xtype, const void * value, size_t len)
{
    checkErr( nc_put_att(nc_id, NC_GLOBAL, name.c_str(), xtype, len, value), "nc_put_att" );
}

void netcdf_interface::put_attr(const std::string & name, const std::vector<float> & attribute, nc_type type)
{
    checkErr( nc_put_att_float(nc_id, NC_GLOBAL, name.c_str(), type, attribute.size(), attribute.data()), "nc_put_att_float" );
}

void netcdf_interface::put_attr(const std::string & name, const std::vector<double> & attribute, nc_type type)
{
    checkErr( nc_put_att_double(nc_id, NC_GLOBAL, name.c_str(), type, attribute.size(), attribute.data()), "nc_put_att_double" );
}

void netcdf_interface::put_attr(const std::string & name, const std::vector<int> & attribute, nc_type type)
{
    checkErr( nc_put_att_int(nc_id, NC_GLOBAL, name.c_str(), type, attribute.size(), attribute.data()), "nc_put_att_int" );
}

void netcdf_interface::put_attr(const std::string & name, const std::vector<unsigned> & attribute, nc_type type)
{
    checkErr( nc_put_att_uint(nc_id, NC_GLOBAL, name.c_str(), type, attribute.size(), attribute.data()), "nc_put_att_uint" );
}

void netcdf_interface::put_attr(const std::string & name, const std::vector<long long> & attribute, nc_type type)
{
    checkErr( nc_put_att_longlong(nc_id, NC_GLOBAL, name.c_str(), type, attribute.size(), attribute.data()), "nc_put_att_longlong" );
}

void netcdf_interface::put_attr(const std::string & name, const std::vector<unsigned long long> & attribute, nc_type type)
{
    checkErr( nc_put_att_ulonglong(nc_id, NC_GLOBAL, name.c_str(), type, attribute.size(), attribute.data()), "nc_put_att_ulonglong" );
}

void netcdf_interface::put_attr(const std::string & name, const std::vector<std::string> & attribute)
{
    std::vector<const char * > tempstrv(attribute.size());
    for(size_t i=0;i<attribute.size();++i)
        tempstrv[i]=attribute[i].c_str();

    checkErr( nc_put_att_string(nc_id, NC_GLOBAL, name.c_str(), tempstrv.size(), tempstrv.data()), "nc_put_att_string" );
}


std::vector<int> netcdf_interface::def_dimensions(const std::vector<std::string> & dimensionnames, const std::vector<size_t> & dimensionsizes)
{
    const size_t ndimensions = dimensionnames.size();

    if(ndimensions!=dimensionsizes.size())
        throw std::runtime_error("netcdf_interface::def_dimensions: dimension size mismatch");

    std::vector<int> dimids(ndimensions);

    for(size_t i = 0; i<ndimensions;++i)
    {
        checkErr( nc_def_dim(nc_id, dimensionnames[i].c_str(), dimensionsizes[i], dimids.data()+i ),"nc_def_dim");
    }


    return dimids;
}

int netcdf_interface::def_variable(const std::string & name, nc_type xtype ,const std::vector<int> & dimids)
{
    int varid;
    checkErr( nc_def_var(nc_id, name.c_str(), xtype, dimids.size(), dimids.data(), &varid ) , "nc_def_var");

    if(deflate_level)
        checkErr( nc_def_var_deflate(nc_id, varid, 1, 1, deflate_level), "nc_def_var_deflate");

    return varid;
}

int netcdf_interface::def_variable(const std::string & name, nc_type xtype ,int dimid)
{
    std::vector dimids(1,dimid);
    return def_variable(name, xtype, dimids);
}

void netcdf_interface::enddef()
{
    checkErr(nc_enddef(nc_id),"nc_eddef");
}

int netcdf_interface::put_variable(const std::string & name, const void * data)
{
    checkErr( nc_put_var(nc_id, get_variable_id(name), data), "nc_put_var" );
    return get_variable_id(name);
}

int netcdf_interface::put_variable(int varid, const void * data)
{
    checkErr( nc_put_var(nc_id, varid, data), "nc_put_var" );
    return varid;
}

int netcdf_interface::put_variable_auto(int varid, const float * data)
{
    checkErr( nc_put_var_float(nc_id, varid, data), "nc_put_var_float" );
    return varid;
}

int netcdf_interface::put_variable_auto(int varid, const double * data)
{
    checkErr( nc_put_var_double(nc_id, varid, data), "nc_put_var_double" );
    return varid;
}

int netcdf_interface::put_variable_auto(int varid, const int * data)
{
    checkErr( nc_put_var_int(nc_id, varid, data), "nc_put_var_int" );
    return varid;
}

int netcdf_interface::put_variable_auto(int varid, const long long * data)
{
    checkErr( nc_put_var_longlong(nc_id, varid, data), "nc_put_var_longlong" );
    return varid;
}

int netcdf_interface::put_variable_auto(int varid, const unsigned * data)
{
    checkErr( nc_put_var_uint(nc_id, varid, data), "nc_put_var_uint" );
    return varid;
}

int netcdf_interface::put_variable_auto(int varid, const unsigned long long * data)
{
    checkErr( nc_put_var_ulonglong(nc_id, varid, data), "nc_put_var_ulonglong" );
    return varid;
}

template<typename T>
int netcdf_interface::put_variable_auto(const std::string & name, const T * data)
{
    return put_variable_auto(get_variable_id(name), data);
}


int netcdf_interface::put_vara(const std::string & name, const std::vector<size_t> & startp, const std::vector<size_t> & countp, const void * data)
{
    checkErr( nc_put_vara(nc_id, get_variable_id(name), startp.data(), countp.data(), data), "nc_put_vara" );
    return get_variable_id(name);
}

void netcdf_interface::close_file()
{
    checkErr(nc_close(nc_id),"nc_close");
}

void netcdf_interface::open_file(int cmode)
{
    checkErr(nc_open(filename.c_str(), cmode, &nc_id),"nc_open");
}

void netcdf_interface::open_file(const std::string & filename_, int cmode)
{
    filename=filename_;
    checkErr(nc_open(filename.c_str(), cmode, &nc_id),"nc_open");
}

void netcdf_interface::get_attr(const std::string & name, float * attribute)
{
    checkErr(nc_get_att_float(nc_id, NC_GLOBAL, name.c_str(), attribute), "nc_get_att_float");
}

void netcdf_interface::get_attr(const std::string & name, double * attribute)
{
    checkErr(nc_get_att_double(nc_id, NC_GLOBAL, name.c_str(), attribute), "nc_get_att_double");
}

void netcdf_interface::get_attr(const std::string & name, int * attribute)
{
    checkErr(nc_get_att_int(nc_id, NC_GLOBAL, name.c_str(), attribute), "nc_get_att_int");
}

void netcdf_interface::get_attr(const std::string & name, unsigned int * attribute)
{
    checkErr(nc_get_att_uint(nc_id, NC_GLOBAL, name.c_str(), attribute), "nc_get_att_uint");
}

void netcdf_interface::get_attr(const std::string & name, long long * attribute)
{
    checkErr(nc_get_att_longlong(nc_id, NC_GLOBAL, name.c_str(), attribute), "nc_get_att_longlong");
}

void netcdf_interface::get_attr(const std::string & name, unsigned long long * attribute)
{
    checkErr(nc_get_att_ulonglong(nc_id, NC_GLOBAL, name.c_str(), attribute), "nc_get_att_ulonglong");
}

void netcdf_interface::get_attr(const std::string & name, void * value )
{
    checkErr(nc_get_att(nc_id, NC_GLOBAL, name.c_str(), value), "nc_get_att");
}

void netcdf_interface::get_attr(const std::string & name, std::string & attribute)
{
    size_t attlen_str = get_attlen(name)+1;
    std::vector<char> intermediate(attlen_str);
    checkErr(nc_get_att_text(nc_id, NC_GLOBAL, name.c_str(), intermediate.data()), "nc_get_att_text");
    intermediate[attlen_str] = '\0';
    attribute.assign(intermediate.data());
}

size_t netcdf_interface::get_attlen(const std::string & name)
{
    size_t attlen;
    checkErr(nc_inq_attlen(nc_id,NC_GLOBAL,name.c_str(),&attlen), "nc_inq_attlen");
    return attlen;
}

void netcdf_interface::get_attr(const std::string & name, std::vector<float> & attribute)
{
    attribute.resize(get_attlen(name));
    get_attr(name,attribute.data());
}

void netcdf_interface::get_attr(const std::string & name, std::vector<double> & attribute)
{
    attribute.resize(get_attlen(name));
    get_attr(name,attribute.data());
}
void netcdf_interface::get_attr(const std::string & name, std::vector<int> & attribute)
{
    attribute.resize(get_attlen(name));
    get_attr(name,attribute.data());
}

void netcdf_interface::get_attr(const std::string & name, std::vector<unsigned> & attribute)
{
    attribute.resize(get_attlen(name));
    get_attr(name,attribute.data());
}

void netcdf_interface::get_attr(const std::string & name, std::vector<long long> & attribute)
{
    attribute.resize(get_attlen(name));
    get_attr(name,attribute.data());
}

void netcdf_interface::get_attr(const std::string & name, std::vector<unsigned long long> & attribute)
{
    attribute.resize(get_attlen(name));
    get_attr(name,attribute.data());
}


void netcdf_interface::get_attr(const std::string & name, std::vector<std::string> & attribute)
{
    attribute.resize(get_attlen(name));

    std::vector<char*> intermediate(get_attlen(name),nullptr);
    checkErr(nc_get_att_string(nc_id, NC_GLOBAL, name.c_str(), intermediate.data()), "nc_get_att_string");

    for(size_t i=0;i<attribute.size();++i)
        attribute[i].assign(intermediate[i]);

    checkErr(nc_free_string(intermediate.size(), intermediate.data()), "nc_free_string");
}

nc_type netcdf_interface::get_attr_type(const std::string & name)
{
    nc_type type;
    checkErr(nc_inq_atttype(nc_id, NC_GLOBAL, name.c_str(), &type), "nc_inq_atttype");
    return type;
}

int netcdf_interface::get_variable(const std::string & name, void * data)
{
    return get_variable(get_variable_id(name), data);
}
int netcdf_interface::get_variable(int varid, void * data)
{
    checkErr(nc_get_var(nc_id, varid, data), "nc_get_var");
    return varid;
}

int netcdf_interface::get_variable_auto(int varid, float * data)
{
    checkErr(nc_get_var_float(nc_id, varid, data), "nc_get_var_float");
    return varid;
}

int netcdf_interface::get_variable_auto(int varid, double * data)
{
    checkErr(nc_get_var_double(nc_id, varid, data), "nc_get_var_double");
    return varid;
}

int netcdf_interface::get_variable_auto(int varid, int * data)
{
    checkErr(nc_get_var_int(nc_id, varid, data), "nc_get_var_int");
    return varid;
}

int netcdf_interface::get_variable_auto(int varid, long long * data)
{
    checkErr(nc_get_var_longlong(nc_id, varid, data), "nc_get_var_longlong");
    return varid;
}

int netcdf_interface::get_variable_auto(int varid, unsigned * data)
{
    checkErr(nc_get_var_uint(nc_id, varid, data), "nc_get_var_uint");
    return varid;
}

int netcdf_interface::get_variable_auto(int varid, unsigned long long * data)
{
    checkErr(nc_get_var_ulonglong(nc_id, varid, data), "nc_get_var_ulonglong");
    return varid;
}

template<typename T>
int netcdf_interface::get_variable_auto(const std::string & name, T * data)
{
    return get_variable_auto(get_variable_id(name), data);
}


int netcdf_interface::get_variable_string(int varid, char ** data)
{
    checkErr(nc_get_var_string(nc_id, varid, data), "nc_get_var_string");
    return varid;
}

template<typename T>
int netcdf_interface::get_variable_vector(const std::string & name, std::vector<T> & variable)
{
    int varid = get_variable_id(name);
    variable.resize(get_variable_size(varid));
    return get_variable_auto(varid,variable.data());
}

int netcdf_interface::get_vara(const std::string & name, const std::vector<size_t> & startp, const std::vector<size_t> & countp, void * data)
{
    checkErr( nc_get_vara(nc_id, get_variable_id(name), startp.data(), countp.data(), data), "nc_get_vara" );
    return get_variable_id(name);
}

nc_type netcdf_interface::get_variable_type(const std::string & name)
{
    return get_variable_type(get_variable_id(name));
}

nc_type netcdf_interface::get_variable_type(int varid)
{
    nc_type type;
    checkErr(nc_inq_vartype(nc_id, varid, &type), "nc_inq_vartype");
    return type;
}

size_t netcdf_interface::get_variable_size(const std::string & name)
{
    return get_variable_size(get_variable_id(name));
}

size_t netcdf_interface::get_variable_size(int varid)
{
    int ndims;
    checkErr(nc_inq_varndims(nc_id,varid,&ndims),"nc_inq_varndims");

    std::vector<int> dimids(ndims);
    checkErr(nc_inq_vardimid(nc_id,varid,dimids.data()),"nc_inq_vardimid");

    size_t varsize = 1;
    size_t dimlen;

    for(const auto & dimid: dimids)
    {
        checkErr(nc_inq_dimlen(nc_id,dimid,&dimlen),"nc_inq_dimlen");
        varsize*=dimlen;
    }

    return varsize;
}

std::vector<size_t> netcdf_interface::get_variable_sizes(const std::string & name)
{
    return get_variable_sizes(get_variable_id(name));
}

std::vector<size_t> netcdf_interface::get_variable_sizes(int varid)
{
    int ndims;
    checkErr(nc_inq_varndims(nc_id,varid,&ndims),"nc_inq_varndims");

    std::vector<int> dimids(ndims);
    checkErr(nc_inq_vardimid(nc_id,varid,dimids.data()),"nc_inq_vardimid");

    size_t dimlen;

    std::vector<size_t> result;
    result.reserve(ndims);

    for(const auto & dimid: dimids)
    {
        checkErr(nc_inq_dimlen(nc_id,dimid,&dimlen),"nc_inq_dimlen");
        result.push_back(dimlen);
    }

    return result;
}

std::vector<std::string> netcdf_interface::get_variable_dimnames(int varid)
{
    int ndims;
    checkErr(nc_inq_varndims(nc_id,varid,&ndims),"nc_inq_varndims");

    std::vector<int> dimids(ndims);
    checkErr(nc_inq_vardimid(nc_id,varid,dimids.data()),"nc_inq_vardimid");

    char dimname[NC_MAX_NAME+1];

    std::vector<std::string> result;
    result.reserve(ndims);

    for(const auto & dimid: dimids)
    {
        checkErr(nc_inq_dimname(nc_id,dimid,dimname),"nc_inq_dimname");
        result.push_back(dimname);
    }

    return result;
}


int netcdf_interface::get_variable_id(const std::string & name)
{
    int varid;
    checkErr(nc_inq_varid(nc_id, name.c_str(), &varid),"nc_inq_varid");
    return varid;
}
int netcdf_interface::get_id()
{
    return nc_id;
}

bool netcdf_interface::variable_defined(const std::string & name)
{
    int varid;
    int err = nc_inq_varid(nc_id, name.c_str(), &varid);
    if(err==NC_NOERR)
        return true;
    else if(err==NC_ENOTVAR)
        return false;
    else
        checkErr(err,"nc_inq_varid");

    return false;
}


// Explicit instantiation
#define NETCDF_EXPLICIT_INSTANTIATE_PUT(type) \
        template int netcdf_interface::put_variable_auto(const std::string & name, const type * data);

NETCDF_EXPLICIT_INSTANTIATE_PUT(float)
NETCDF_EXPLICIT_INSTANTIATE_PUT(double)
NETCDF_EXPLICIT_INSTANTIATE_PUT(int)
NETCDF_EXPLICIT_INSTANTIATE_PUT(unsigned int)
NETCDF_EXPLICIT_INSTANTIATE_PUT(long long)
NETCDF_EXPLICIT_INSTANTIATE_PUT(unsigned long long)

#undef NETCDF_EXPLICIT_INSTANTIATE_PUT

#define NETCDF_EXPLICIT_INSTANTIATE_GET(type) \
        template int netcdf_interface::get_variable_auto(const std::string & name, type * data);\
        template int netcdf_interface::get_variable_vector(const std::string & name, std::vector<type> & data);

NETCDF_EXPLICIT_INSTANTIATE_GET(float)
NETCDF_EXPLICIT_INSTANTIATE_GET(double)
NETCDF_EXPLICIT_INSTANTIATE_GET(int)
NETCDF_EXPLICIT_INSTANTIATE_GET(unsigned int)
NETCDF_EXPLICIT_INSTANTIATE_GET(long long)
NETCDF_EXPLICIT_INSTANTIATE_GET(unsigned long long)

#undef NETCDF_EXPLICIT_INSTANTIATE_GET
