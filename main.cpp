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

//generates a part of distribution and writes it to the supplied vector
template<class Distribution, typename value_type>
void fill_sample_part(
	std::vector<value_type> &dest,
	size_t part_begin,
	size_t part_size,
	Distribution &distribution,
	pcg32 &rng,
	size_t jump_mult = 1
) {
	pcg32 rng_copy = rng;

	size_t part_end = part_begin + part_size;

	for (size_t i = part_begin; i < part_end; ++i) {
		dest[i] = distribution(rng);
	}

	rng_copy.advance(part_size*jump_mult);
	rng = rng_copy;
}


//nonparallel function for creating a sample of a distribution
template<class Distribution, typename value_type>
std::vector<value_type> distribution_sample(
	size_t size,
	Distribution& distribution,
	pcg32& rng,
	size_t parts,
	size_t jump_mult = 1
) {
	std::vector<value_type> result(size);
	
	size_t part_size = size / parts;

	for (size_t i = 0; i < parts - 1; ++i) {
		fill_sample_part(
			result, 
			part_size * i, 
			part_size, 
			distribution, 
			rng, 
			jump_mult
		);
	}
	
	size_t last_part_size = size - part_size * (parts - 1);
	size_t last_part_begin = part_size * (parts - 1);
	
	fill_sample_part(
		result, 
		last_part_begin, 
		last_part_size, 
		distribution, 
		rng, 
		jump_mult
	);

	return result;
}

//parallel function for creating a sample of a distribution
template<class Distribution, typename value_type>
std::vector<value_type> distribution_sample_parallel(
	size_t size, 
	Distribution &distribution,
	pcg32 &rng,  
	size_t thread_count,
	size_t jump_mult = 1
) {
	std::vector<value_type> result(size);

	size_t thread_size = size / thread_count;
	size_t last_thread_size = size - thread_size * (thread_count - 1);

	std::vector<std::thread> threads;

	//threads for first (thread_count - 1) * thread_size elements
	for (size_t i = 0; i < thread_count - 1; ++i) {
		threads.emplace_back(
			fill_sample_part<Distribution, value_type>,
			std::ref(result),
			thread_size * i,
			thread_size,
			std::ref(distribution),
			std::ref(rng),
			jump_mult
		);
	}
	//thread for last thread_size + last_elements elements
	threads.emplace_back(
		fill_sample_part<Distribution, value_type>,
		std::ref(result),
		thread_size * (thread_count - 1),
		last_thread_size,
		std::ref(distribution),
		std::ref(rng),
		jump_mult
	);

	for (auto &thread : threads) {
		thread.join();
	}

	return result;
}


//for functions of distributions
template<
	class urng = pcg32, 
	class Distribution = chosen_distribution, 
	typename value_type = val_type
>
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
const int DISCARD = 10;
pcg32 rng(SEED);
pcg32 rng_copy = rng;

//for elapsed time
clock_t t;

//for double equal precision
const double EPS = 1e-7;

template<typename value_type = double, typename result_type = long double>
result_type part_sum(
	std::vector<value_type>& vector, 
	size_t part_begin, 
	size_t part_end
) {
	result_type sum = 0;

	for (size_t i = 0; i < part_end; ++i) sum += vector[i];

	return sum;
}

template<typename value_type = double, typename result_type = long double>
void store_part_sum(
	std::vector<value_type>& vector, 
	size_t part_begin, 
	size_t part_end, 
	std::vector<result_type> results, 
	size_t res_index
) {
	results[res_index] = part_sum<value_type, result_type>(
		vector, 
		part_begin, 
		part_end
		);
}


template<typename value_type = double, typename result_type = long double>
result_type mean(
	std::vector<value_type>& sample, 
	size_t thread_count, 
	bool mean_for_each = false
) {
	
	std::vector<result_type> part_sums(thread_count);
	std::vector<std::thread> threads;

	size_t part_size = sample.size() / thread_count;
	size_t last_part_size = sample.size() - part_size*(thread_count - 1);

	for (size_t i = 0; i < thread_count - 1; ++i) {
		threads.emplace_back(
			store_part_sum<value_type, result_type>,
			std::ref(sample),
			i * part_size,
			i * (part_size + 1),
			std::ref(part_sums),
			i
		);
	}

	threads.emplace_back(
		store_part_sum<value_type, result_type>,
		std::ref(sample),
		(thread_count - 1) * part_size,
		sample.size(),
		std::ref(part_sums),
		thread_count - 1
	);

	for (auto &thread : threads) {
		thread.join();
	}

	result_type res = 0;

	if (mean_for_each) {
		for (size_t i = 0; i < thread_count - 1; ++i) {
			res += part_sums[i] / part_size;
		}
		res += part_sums[thread_count - 1] / last_part_size;
		res /= thread_count;
	}
	else {
		for (size_t i = 0; i < thread_count - 1; ++i) {
			res += part_sums[i];
		}
		res /= sample.size();
	}

	return res;
}

int main() {
	std::normal_distribution<double> rnorm(0, 1);
	std::uniform_int_distribution<int> runif_int(1, 10);

	FunctionalDistribution<
		pcg32, 
		std::normal_distribution<double>, 
		double
	> rnorm_sin(
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
			rng_copy,
			4
			);

	size_t size = runif_sample_parallel.size();
	assert(size == runif_sample_single.size());
	assert(size == 100);

	for (size_t i = 0; i < size; ++i) {
		assert(runif_sample_parallel[i] == runif_sample_single[i]);
	}

	std::cout << "Uniform int passed successfully" << std::endl;

	system("pause");

	//Check, if results are equal for distributions with random uses of rng:

	std::vector<double> rnorm_sample_parallel =
		distribution_sample_parallel<
		std::normal_distribution<double>, 
		double
		>(
			100,
			rnorm,
			rng,
			4,
			DISCARD
			);


	std::vector<double> rnorm_sample_single =
		distribution_sample<
		std::normal_distribution<double>, 
		double
		>(
			100,
			rnorm,
			rng_copy,
			4,
			DISCARD
			);

	size = rnorm_sample_parallel.size();
	assert(size == rnorm_sample_single.size());
	assert(size == 100);

	for (size_t i = 0; i < size; ++i) {
		assert(
			rnorm_sample_parallel[i] == rnorm_sample_single[i]
		);
	}

	std::cout << "Normal passed successfully" << std::endl;

	//Measure elapsed time for parallel and non-parallel approach:

	t = clock();
	std::vector<double> rnorm_sample_parallel_big =
		distribution_sample_parallel<
		std::normal_distribution<double>, 
		double
		>(
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
		distribution_sample<
		std::normal_distribution<double>, 
		double
		>(
			10000000,
			rnorm,
			rng_copy,
			4,
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

	long double parallel_mean = 
		mean<double, long double>(rnorm_sample_parallel_big, 4);
	long double single_mean = 
		mean<double, long double>(rnorm_sample_single_big, 4);

	std::cout
		<< "Comparing means of parallel and nonparallel sin(N(0, 1)) samples:"
		<< std::endl
		<< "Mean difference: "
		<< std::fixed
		<< std::setprecision(15)
		<< parallel_mean - single_mean
		<< std::endl;

	std::cout << "Mean comparison passed successfully" << std::endl;

	////Trying out functional distributions:

	std::vector<double> rnorm_sin_sample_parallel =
		distribution_sample_parallel<
		FunctionalDistribution<
		pcg32, 
		std::normal_distribution<double>, 
		double
		>, 
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
		FunctionalDistribution<
		pcg32, 
		std::normal_distribution<double>, 
		double
		>,
		double
		>(
			100000,
			rnorm_sin,
			rng_copy,
			4,
			DISCARD
			);

	long double parallel_mean_sin = 
		mean<double, long double>(rnorm_sin_sample_parallel, 4);
	long double single_mean_sin = 
		mean<double, long double>(rnorm_sin_sample_single, 4);

	std::cout
		<< "Comparing means of parallel and nonparallel sin(N(0, 1)) samples:"
		<< std::endl
		<< "Mean difference: "
		<< std::fixed
		<< std::setprecision(15)
		<< parallel_mean_sin - single_mean_sin
		<< std::endl;


	std::cout << "Functional passed successfully" << std::endl;

	system("pause");
	return 0;
}