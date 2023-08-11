#include <vector>
#include <cmath>
#include <utility>
#include <exception>
#include <stdexcept>
#include <algorithm>

template<typename T>
class WelfordMeanVariance {

    private:

    std::vector<T> mean;
    std::vector<T> sum_squares;
    unsigned int len;
    unsigned int n;

    public:

    WelfordMeanVariance(unsigned int _len): len(_len), n(0)
    {
        mean.resize(_len);
        sum_squares.resize(_len);
        std::fill(mean.begin(), mean.end(), static_cast<T>(0));
        std::fill(sum_squares.begin(), sum_squares.end(), static_cast<T>(0));
    }
    
    void update(std::vector<T> &x)
    {
        if(x.size() != len)
            throw std::length_error("Welford update: length mismatch.");

        n++;

        for(unsigned int i = 0; i < len; i++)
        {
            T delta = x[i] - mean[i];
            mean[i] += delta / T(n);
            T delta2 = x[i] - mean[i];
            sum_squares[i] += delta * delta2;
        }
    }

    std::pair<std::vector<T>,std::vector<T>> getMeanVariance()
    {
        std::vector<T> variance(len);
        std::transform(sum_squares.begin(), sum_squares.end(), variance.begin(),
            [N=n](const T& x) { return x/T(N); });

        return std::make_pair(mean,variance);
    }
};
