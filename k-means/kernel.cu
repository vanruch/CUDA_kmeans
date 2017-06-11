#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include "device_functions.h"


#include <cuda_runtime_api.h>

#include <device_launch_parameters.h>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string>
#include <numeric>
#include <ctime>

#include <thrust\host_vector.h>
#include <thrust\device_vector.h>
#include <thrust\sort.h>
#include <thrust\copy.h>
#include <thrust\device_ptr.h>
#include <thrust\device_malloc.h>
#include <thrust\for_each.h>
#include <thrust\transform.h>
#include <thrust/execution_policy.h>
#include <thrust\scan.h>

#define SIZE 50000
#define K 3
#define LO -1000.0
#define HI 1000.0

using namespace std;
using namespace thrust;

void readData(host_vector<float3> &vec) {
	
	for (int i = 0; i < SIZE; i++)
	{
		float3 f;
		f.x= LO + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (HI - LO)));
		f.y= LO + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (HI - LO)));
		f.z= LO + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (HI - LO)));
		vec.push_back(f);
	}

}

struct dist_functor {
	float3* centr;
	
	dist_functor(float3* _centr) : centr(_centr) {}

	__host__ __device__ int operator()(const float3 point) {
		float dist = INFINITY;
		int r = 0;
		for (int i = 0; i < K; i++)
		{
			int d = (centr[i].x - point.x)*(centr[i].x - point.x)+
				(centr[i].y - point.y)*(centr[i].y - point.y)+
				(centr[i].z - point.z)*(centr[i].z - point.z);
			if (d < dist) {
				r = i;
				dist = d;
			}
		}
		return r;
	}
};

struct sum_functor {
	float3* centr;
	int* sums;

	sum_functor(float3* _centr, int* _sums) : centr(_centr), sums(_sums) {}

	 __device__ void operator()(const thrust::tuple<float3, int> &point) {
		int i = point.get<1>();
		atomicAdd(&centr[i].x, point.get<0>().x);
		atomicAdd(&centr[i].y, point.get<0>().y);
		atomicAdd(&centr[i].z, point.get<0>().z);
		atomicAdd(&sums[i], 1);
	}
};

struct mean_functor
{
	mean_functor()
	{

	}
	__host__ __device__ float3 operator()(float3 p, int c) {
		float3 t;
		t.x = p.x / (float)c;
		t.y = p.y / (float)c;
		t.z = p.z / (float)c;
		return t;
	}

};

struct eq_functor {

	__host__ __device__ bool operator()(const thrust::tuple<int,int> &t) {
		return  t.get<1>() != t.get<0>();
	}
};

host_vector<int> cudaKMeans(host_vector<float3> &points) {
	device_vector<float3> d_points = points;
	host_vector<float3> centroids;
	for (int i = 0; i < K; i++)
	{
		centroids.push_back(points[i]);
	}
	device_vector<float3> d_centroids = centroids;
	device_vector<int> d_indexes(points.size(), -1);
	device_vector<int> d_new_indexes(points.size());
	device_vector<int> cnt(K);
	int diff=1;
	while (diff>0)
	{
		thrust::transform(d_points.begin(), d_points.end(), d_new_indexes.begin(), dist_functor(raw_pointer_cast(d_centroids.data())));
		diff = thrust::count_if(make_zip_iterator(thrust::make_tuple(d_indexes.begin(), d_new_indexes.begin())),
			make_zip_iterator(thrust::make_tuple(d_indexes.end(), d_new_indexes.end())), eq_functor());		
		thrust::copy(d_new_indexes.begin(), d_new_indexes.end(), d_indexes.begin());
		thrust::fill(cnt.begin(), cnt.end(), 0);
		thrust::fill(d_centroids.begin(), d_centroids.end(), float3());
		thrust::for_each(make_zip_iterator(thrust::make_tuple(d_points.begin(), d_indexes.begin())),
			make_zip_iterator(thrust::make_tuple(d_points.end(), d_indexes.end())),
			sum_functor(raw_pointer_cast(d_centroids.data()), raw_pointer_cast(cnt.data())));
		thrust::transform(d_centroids.begin(), d_centroids.end(), cnt.begin(), d_centroids.begin(), mean_functor());
	}
	return host_vector<int>(d_indexes);
}

//--------------HOST------------------
void new_ind(host_vector<float3> &centr, host_vector<float3> &points, host_vector<int> &indexes) {
	for (int j = 0; j < SIZE; j++)
	{
		float dist = INFINITY;

		for (int i = 0; i < K; i++)
		{
			int d = (centr[i].x - points[j].x)*(centr[i].x - points[j].x) +
				(centr[i].y - points[j].y)*(centr[i].y - points[j].y) +
				(centr[i].z - points[j].z)*(centr[i].z - points[j].z);
			if (d < dist) {
				indexes[j] = i;
				dist = d;
			}
		}
	}
}

host_vector<int> hostKMeans(host_vector<float3> &points) {
	host_vector<float3> centroids;
	for (int i = 0; i < K; i++)
	{
		centroids.push_back(points[i]);
	}
	host_vector<int> indexes(points.size(), -1);
	host_vector<int> new_indexes(points.size());
	int diff = 1;
	while (diff > 0) {
		new_ind(centroids, points, new_indexes);
		diff = 0;
		for (int i = 0; i < SIZE; i++)
		{
			if (new_indexes[i] != indexes[i]) {
				indexes[i] = new_indexes[i];
				diff++;
			}
		}
		host_vector<int> cnt(K);
		for (size_t i = 0; i < K; i++)
		{
			centroids[i] = float3();
		}
		for (int i = 0; i < SIZE; i++)
		{
			cnt[indexes[i]]++;
			centroids[indexes[i]].x += points[i].x;
			centroids[indexes[i]].y += points[i].y;
			centroids[indexes[i]].z += points[i].z;
		}
		for (size_t i = 0; i < K; i++)
		{
			centroids[i].x /= cnt[i];
			centroids[i].y /= cnt[i];
			centroids[i].z /= cnt[i];
		}
	}
	return indexes;
}


int main()
{


	
	host_vector<float3> points;



	cout << "gen\n";
	readData(points);
	cout << "calc 1\n";
	clock_t begin = clock();
	auto v = cudaKMeans(points);
	clock_t end = clock();
	cout << "CUDA: " << double(end - begin) / CLOCKS_PER_SEC << endl;
	
	begin = clock();
	auto v2 = hostKMeans(points);
	end = clock();
	cout << "HOST: " << double(end - begin) / CLOCKS_PER_SEC << endl;

	for (int i = 0; i < SIZE; i++)
	{
		if (v[i] != v2[i]) {
			cout << "ERR";
			return 0;
		}
	}
	int a[K] = { 0 };
	ofstream myfile;
	myfile.open("kmeans.csv");
	myfile << "X,Y,Z,cluster\n";
	

	for (int i = 0; i < SIZE; i++)
	{

		a[v[i]]++;
		myfile << points[i].x << ',' << points[i].y << ',' << points[i].z << ',' << v[i] << endl;
	}
	myfile.close();
	for (size_t i = 0; i < K; i++)
	{
		cout << a[i] << endl;
	}
	return 0;
}