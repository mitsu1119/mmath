#pragma once
#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include "util.cuh"

namespace mmath {

// 長さlenの16進文字列を数値にデコードしてresに格納
__host__ __device__ void small_hex_str_to_i64(const char *st, i32 len, i64 *res);

// 16進数のデコード
// 長さlenのstをsizeごとに切り分け、各部分文字列を16進数の数値と見て、サイズNのdataに数値として格納
__global__ void decode_hex_for_digits(const char *st, size_t len, i32 size, i64 *data, size_t N);

class Digits {
private:
	// Digits(123456789) => data: {56789, 1234}
	thrust::device_vector<i64> data;

public:
	Digits(i64 val = 0);	

	// dataをlenの長さのvalで埋める
	Digits(size_t len, i64 val);

	// log2(digitsの基数)、つまり基数は2の何乗かを示す
	// 16進数への変換を考えて、LOG_RADIX % 4 = 0 である必要あり
	static constexpr i32 LOG_RADIX = 20;

	// 16進数として見たときの各桁の長さ
	static constexpr i32 LOG_16_RADIX = mmath::Digits::LOG_RADIX >> 2;

	size_t size() const;
	i64 at(size_t i) const;
	size_t msd() const;

	void to_zero();		// dataを長さ1のゼロの配列にする
	void push_msd(i64 num);
	void push_lsd_zero(size_t n);
	void pop_msd();
	void resize(size_t size);
	void normalize();	// 不必要な上位桁のゼロを削除する

	// 文字列を16進数と解釈して格納
	void from_hex(const char *st, size_t len_st);

	void print(bool hex = false) const;
};

inline mmath::Digits::Digits(size_t len, i64 val): data(len, val) {
}

inline mmath::Digits::Digits(i64 val): Digits(1, val) {
}

inline size_t mmath::Digits::size() const {
	return data.size();
}

inline i64 mmath::Digits::at(size_t i) const {
	return data[i];
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

}
