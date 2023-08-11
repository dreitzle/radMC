#define BOOST_TEST_MODULE welford_algorithm
#include <boost/test/unit_test.hpp>

#include "welford.tcc"

/* Required precision (double) in percent*/
#define PREC 5e-12

/* Test Legendre algorithms with double precision */
BOOST_AUTO_TEST_SUITE( welford_double )

    /* Test construction */
    BOOST_AUTO_TEST_CASE( welford_double_construct )
    {
        BOOST_REQUIRE_NO_THROW( WelfordMeanVariance<double>(1) );
    }

    /* Test double vector */
    BOOST_AUTO_TEST_CASE( welford_double )
    {
        std::vector<double> v0 = {0.1043298428188815,1.184465406140853,80.56790212532181,99.00067502636573,0.5701916087455388,
                                  5.068875108261159,1.989565039724661,-3.346552602322213,7.124141745800378,1.369568471357605};
        std::vector<double> v1 = {1.35390790994269,8.893051339003614,74.59276285691924,99.86154640065973,-0.3823344911033815,
                                  5.09792870099476,2.295626350885145,0.04756987824128777,8.496625322599783,3.56686630324396};
        std::vector<double> v2 = {0.7124566615881187,5.352216307781481,11.02913662395661,99.41299742454856,0.4320083525522498,
                                  5.09074515849136,3.414559836231516,1.173590762951299,10.20510007806256,-2.809205770352432};
        std::vector<double> v3 = {0.132658175132506,1.139029582621416,68.51953364000246,99.22702819447753,0.1915003027960753,
                                  5.048410874431594,0.1765675708142303,0.9053908072721076,4.802396716656183,3.963553516507252};
        std::vector<double> v4 = {0.9517940925318902,7.788094858325898,88.30141623519583,99.56156015234436,0.6291953799039911,
                                  5.0846048872752,0.3247134633924582,-1.032688166173153,7.444495650398617,0.9111626031557784};
        std::vector<double> v5 = {1.250490944453572,5.553963389696481,87.50172182214446,99.7936270863909,0.1726797796232129,
                                  5.07711905808399,1.290194666518151,-0.1283882763159665,9.308200598248225,3.024540309614225};
        std::vector<double> v6 = {1.114771164383137,1.19421412518038,71.63498922880672,99.71237975261685,0.8692111631429087,
                                  5.095219334380705,0.2769851842882475,0.9520275649059144,9.326613107745656,1.222571101320842};
        std::vector<double> v7 = {1.028920002050984,9.957615465256314,44.01296776729426,99.4892307966303,-0.3586299341371413,
                                  5.005926453867509,0.8452153829657143,-0.7739711553188853,8.11118444247094,0.06487821867673384};
        std::vector<double> v8 = {1.847325798326465,6.257670003233918,88.93999223613059,99.62723433885866,-0.5274721771538144,
                                  5.015597411857724,0.3578585039448232,2.074383451702273,12.39699120237631,-1.72756994934615};
        std::vector<double> v9 = {0.05751367393974216,2.006904940155899,52.32091477831765,99.97108719934045,0.1283111126788661,
                                  5.09340147044009,0.6554831343534696,0.7467219998054971,13.7142993421917,4.410271832904283};

        std::vector<double> mean_ref = {0.8554168265167987,4.932722541739626,66.74213373140896,99.5657366372233,0.1724661097048505,
                                        5.067782845808409,1.162676913311842,0.06180842647481605,9.093004820655036,1.39966366370821};

        std::vector<double> var_ref = {0.3238715704050343,10.23826967692638,549.6916104884916,0.07922506087850802,0.2001695373932731,
                                       0.001014126021348532,1.051936011996272,2.080949887171554,5.993328893618858,5.252100907120878};

        WelfordMeanVariance<double> welford(10);

        welford.update(v0);
        welford.update(v1);
        welford.update(v2);
        welford.update(v3);
        welford.update(v4);
        welford.update(v5);
        welford.update(v6);
        welford.update(v7);
        welford.update(v8);
        welford.update(v9);

        const auto& [mean, var] = welford.getMeanVariance();

        BOOST_CHECK_CLOSE( mean[0], mean_ref[0] , PREC);
        BOOST_CHECK_CLOSE( mean[1], mean_ref[1] , PREC);
        BOOST_CHECK_CLOSE( mean[2], mean_ref[2] , PREC);
        BOOST_CHECK_CLOSE( mean[3], mean_ref[3] , PREC);
        BOOST_CHECK_CLOSE( mean[4], mean_ref[4] , PREC);
        BOOST_CHECK_CLOSE( mean[5], mean_ref[5] , PREC);
        BOOST_CHECK_CLOSE( mean[6], mean_ref[6] , PREC);
        BOOST_CHECK_CLOSE( mean[7], mean_ref[7] , PREC);
        BOOST_CHECK_CLOSE( mean[8], mean_ref[8] , PREC);
        BOOST_CHECK_CLOSE( mean[9], mean_ref[9] , PREC);

        BOOST_CHECK_CLOSE( var[0], var_ref[0] , PREC);
        BOOST_CHECK_CLOSE( var[1], var_ref[1] , PREC);
        BOOST_CHECK_CLOSE( var[2], var_ref[2] , PREC);
        BOOST_CHECK_CLOSE( var[3], var_ref[3] , PREC);
        BOOST_CHECK_CLOSE( var[4], var_ref[4] , PREC);
        BOOST_CHECK_CLOSE( var[5], var_ref[5] , PREC);
        BOOST_CHECK_CLOSE( var[6], var_ref[6] , PREC);
        BOOST_CHECK_CLOSE( var[7], var_ref[7] , PREC);
        BOOST_CHECK_CLOSE( var[8], var_ref[8] , PREC);
        BOOST_CHECK_CLOSE( var[9], var_ref[9] , PREC);

    }

BOOST_AUTO_TEST_SUITE_END()
