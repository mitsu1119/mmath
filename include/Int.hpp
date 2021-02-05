#pragma once
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include "util.hpp"

namespace mmath {

class Digits {
private:
	// Digits(12345678) => data: {78, 56, 34, 12}
	thrust::device_vector<i64> data;
	
public:
	Digits(i64 val = 0);	

	// dataをlenの長さのvalで埋める
	Digits(size_t len, i64 val);

	size_t size() const;
	size_t msd() const;

	void to_zero();		// dataを長さ1のゼロの配列にする
	void push_msd(i64 num);
	void push_lsd_zero(size_t n);
	void pop_msd();
	void resize(size_t size);
	void normalize();	// 不必要な上位桁のゼロを削除する

	void print() const;
};

inline mmath::Digits::Digits(size_t len, i64 val): data(len, val) {
}

inline mmath::Digits::Digits(i64 val): Digits(1, val) {
}

inline size_t mmath::Digits::size() const {
	return data.size();
}

inline size_t mmath::Digits::msd() const {
	return data.back();
}

inline void mmath::Digits::to_zero() {
	thrust::device_vector<i64>().swap(data);
	data = thrust::device_vector<i64>(1, 0);
}

inline void mmath::Digits::push_msd(i64 num) {
	data.push_back(num);
}

inline void mmath::Digits::push_lsd_zero(size_t n) {
	data.insert(data.begin(), n, 0);
}

inline void mmath::Digits::pop_msd() {
	data.pop_back();
}

inline void mmath::Digits::resize(size_t size) {
	data.resize(size, 0);
}

inline void mmath::Digits::normalize() {
	while(size() > 0) {
		if(msd() == 0) pop_msd();
		else break;
	}

	if(size() == 0) to_zero();
}

void mmath::Digits::print() const {
	for(auto i: data) std::cout << i << " ";
	std::cout << std::endl;
}

}
