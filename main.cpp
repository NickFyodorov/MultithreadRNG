#include <iostream>
#include <iomanip>
#include <thread>
#include <mutex>
#include <vector>
#include <memory>
#include <future>
#include <random>
#include <cstdlib>
#include <time.h>
#include <functional>
#include <cassert>

#include "pcg/pcg_random.hpp"
#include "pcg/pcg_extras.hpp"

template<class Distribution, typename value_type>
value_type new_value_with_discard(
	Distribution &distribution,
	pcg32 &rng,
	size_t discard = 1
) {
	if (discard == 1) return distribution(rng);

	pcg32 rng_copy = rng;
	value_type res = distribution(rng_copy);

	if (rng_copy - rng > discard) throw std::runtime_error("Generator was used more than <discard> times.");

	rng.advance(discard);
	return res;
}

//nonparallel function for creating a sample of a distribution
template<class Distribution, typename value_type>
std::vector<value_type> distribution_sample(
	size_t size,
	Distribution& dist,
	pcg32& rng,
	size_t discard = 1
) {
	std::vector<value_type> result(size);
	for (size_t i = 0; i < size; ++i) {
		result[i] = 
			new_value_with_discard<Distribution, value_type>(
				dist, 
				rng, 
				discard
				);
	}
	return result;
}

//used in distribution_sample_parallel
//generates a part of distribution and writes it to the supplied vector
template<class Distribution, typename value_type>
void fill_sample_part(
	size_t size,
	Distribution &distribution,
	pcg32 &rng,
	size_t steps,
	std::vector<value_type> &dest,
	size_t from,
	size_t discard = 1
) {
	pcg32 rng_copy = rng;
	if(steps > 0) rng_copy.advance(steps * discard);

	size_t end = from + size;

	for (size_t i = from; i < end; ++i) {
		dest[i] = 
			new_value_with_discard<Distribution, value_type>(
			distribution, 
			rng_copy, 
			discard
			);
	}
}

//parallel function for creating a sample of a distribution
template<class Distribution, typename value_type>
std::vector<value_type> distribution_sample_parallel(
	size_t size, 
	Distribution &distribution,
	pcg32 &rng,  
	size_t thread_count,
	size_t discard = 1
) {
	std::vector<value_type> result(size);

	size_t thread_size = size / thread_count;
	size_t last_elements = size - thread_size * thread_count;

	std::vector<std::thread> threads;

	//threads for first (thread_count - 1) * thread_size elements
	for (size_t i = 0; i < thread_count - 1; ++i) {
		threads.emplace_back(
			fill_sample_part<Distribution, value_type>,
			thread_size,
			std::ref(distribution),
			std::ref(rng),
			thread_size * i, //advance generator by another thread_size steps
			std::ref(result),
			thread_size * i, //write after the previous part
			discard
		);
	}
	//thread for last thread_size + last_elements elements
	threads.emplace_back(
		fill_sample_part<Distribution, value_type>,
		thread_size + last_elements,
		std::ref(distribution),
		std::ref(rng),
		thread_size * (thread_count - 1),
		std::ref(result),
		thread_size * (thread_count - 1),
		discard
	);

	for (auto &thread : threads) {
		thread.join();
	}

	return result;
}


//for functions of distributions
template<class urng = pcg32, class Distribution = chosen_distribution, typename value_type = val_type>
class FunctionalDistribution {
private:
	std::shared_ptr<Distribution> distribution;
	std::function<value_type(value_type)> transform;
public:
	FunctionalDistribution(
		std::shared_ptr<Distribution> _distribution,
		std::function<value_type(value_type)> _transform
	) : distribution(_distribution), transform(_transform) {}

	value_type operator()(urng& rng) {
		return transform((*distribution)(rng));
	}

	~FunctionalDistribution() {

	}
};



//for rng
const int SEED = 1;
const int DISCARD = 100;
pcg32 rng(SEED);

//for elapsed time
clock_t t;

//for double equal precision
const double EPS = 1e-7;

template<typename value_type = double, typename result_type = long double>
value_type mean(std::vector<value_type>& sample) {
	result_type res = 0;
	size_t size = sample.size();

	for (size_t i = 0; i < size; ++i) {
		res += sample[i];
	}

	res /= size;

	return res;
}

int main() {
	std::normal_distribution<double> rnorm(0, 1);
	std::uniform_int_distribution<int> runif_int(1, 10);

	FunctionalDistribution<pcg32, std::normal_distribution<double>, double> rnorm_sin(
		std::make_shared<std::normal_distribution<double>>(0, 1),
		[](double input) {return sin(input); }
	);

	//Check, if results are equal for distributions with fixed uses of rng:

	std::vector<int> runif_sample_parallel =
		distribution_sample_parallel<std::uniform_int_distribution<int>, int>(
			100,
			runif_int,
			rng,
			4
			);


	std::vector<int> runif_sample_single =
		distribution_sample<std::uniform_int_distribution<int>, int>(
			100,
			runif_int,
			rng
			);

	size_t size = runif_sample_parallel.size();
	assert(size == runif_sample_single.size());
	assert(size == 100);

	for (size_t i = 0; i < size; ++i) {
		assert(runif_sample_parallel[i] == runif_sample_single[i]);
	}

	std::cout << "Uniform int passed successfully" << std::endl;

	//Check, if results are equal for distributions with random uses of rng:

	std::vector<double> rnorm_sample_parallel =
		distribution_sample_parallel<std::normal_distribution<double>, double>(
			100,
			rnorm,
			rng,
			4,
			DISCARD
			);


	std::vector<double> rnorm_sample_single =
		distribution_sample<std::normal_distribution<double>, double>(
			100,
			rnorm,
			rng,
			DISCARD
			);

	size = rnorm_sample_parallel.size();
	assert(size == rnorm_sample_single.size());
	assert(size == 100);

	for (size_t i = 0; i < size; ++i) {
		assert(
			fabs(rnorm_sample_parallel[i] - rnorm_sample_single[i]) < EPS
		);
	}

	std::cout << "Normal passed successfully" << std::endl;

	//Measure elapsed time for parallel and non-parallel approach:

	t = clock();
	std::vector<double> rnorm_sample_parallel_big =
		distribution_sample_parallel<std::normal_distribution<double>, double>(
			10000000,
			rnorm,
			rng,
			4,
			DISCARD
			);
	t = clock() - t;

	std::cout
		<< "Time of parallel generation: "
		<< std::setw(5)
		<< t * 0.001
		<< " seconds"
		<< std::endl;

	t = clock();
	std::vector<double> rnorm_sample_single_big =
		distribution_sample<std::normal_distribution<double>, double>(
			10000000,
			rnorm,
			rng,
			DISCARD
			);
	t = clock() - t;

	std::cout
		<< "Time of nonparallel generation: "
		<< std::setw(5)
		<< t * 0.001
		<< " seconds"
		<< std::endl;

	std::cout << "Time evaluation passed successfully" << std::endl;

	////Calculating means:

	long double parallel_mean = mean<>(rnorm_sample_parallel_big);
	long double single_mean = mean<>(rnorm_sample_single_big);

	std::cout
		<< "Comparing means of parallel and nonparallel N(0, 1) samples:"
		<< std::endl
		<< "Parallel mean: "
		<< std::fixed
		<< std::setprecision(7)
		<< parallel_mean
		<< std::endl
		<< "Single mean: "
		<< std::fixed
		<< std::setprecision(7)
		<< single_mean
		<< std::endl;


	std::cout << "Mean comparison passed successfully" << std::endl;

	////Trying out functional distributions:

	std::vector<double> rnorm_sin_sample_parallel =
		distribution_sample_parallel<
		FunctionalDistribution<pcg32, std::normal_distribution<double>, double>, 
		double
		>(
			100000,
			rnorm_sin,
			rng,
			4,
			DISCARD
		);

	std::vector<double> rnorm_sin_sample_single =
		distribution_sample<
		FunctionalDistribution<pcg32, std::normal_distribution<double>, double>,
		double
		>(
			100000,
			rnorm_sin,
			rng,
			DISCARD
			);

	long double parallel_mean_sin = mean<>(rnorm_sin_sample_parallel);
	long double single_mean_sin = mean<>(rnorm_sin_sample_single);

	std::cout
		<< "Comparing means of parallel and nonparallel sin(N(0, 1)) samples:"
		<< std::endl
		<< "Parallel mean: "
		<< std::fixed
		<< std::setprecision(7)
		<< parallel_mean_sin
		<< std::endl
		<< "Single mean: "
		<< std::fixed
		<< std::setprecision(7)
		<< single_mean_sin
		<< std::endl;


	std::cout << "Functional passed successfully" << std::endl;

	system("pause");
	return 0;
}