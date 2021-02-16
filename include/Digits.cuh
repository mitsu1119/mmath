#pragma once
#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include "util.cuh"

// add algorithm
// #define MMATH_DIGITS_ADD_SEQUENTIAL 1
#define MMATH_DIGITS_ADD_PARALLEL 1

namespace mmath {

class Digits {
private:
	// Digits(123456789) => data: {56789, 1234}
	thrust::device_vector<i64> data;

	// dataをlenの長さのvalで埋める
	Digits(size_t len, i64 val);

public:
	Digits(i64 val = 0);	


	// log2(digitsの基数)、つまり基数は2の何乗かを示す
	// 16進数への変換を考えて、LOG_RADIX % 4 = 0 である必要あり
	static constexpr i32 LOG_RADIX = 20;

	// 16進数として見たときの各桁の長さ
	static constexpr i32 LOG_16_RADIX = mmath::Digits::LOG_RADIX >> 2;

	// 基数
	static constexpr i64 RADIX = (1 << mmath::Digits::LOG_RADIX);

	size_t size() const;
	i64 at(size_t i) const;
	size_t msd() const;

	// radix-1以外になる最初の要素番号
	size_t first_non_radixmax_index() const;

	void to_zero();		// dataを長さ1のゼロの配列にする
	void push_msd(i64 num, size_t n = 1);
	void push_lsd(i64 num, size_t n = 1);
	void pop_msd();

	void normalize();	// 不必要な上位桁のゼロを削除する

	void add(const mmath::Digits &xx);	// this += x
	void sub(const mmath::Digits &x);	// this -= x, this >= x が前提
	void increment();

	// 文字列を16進数と解釈して格納
	void from_hex(const char *st, size_t len_st);

	void print(bool hex = false) const;
};

namespace Digits_utils {
// 長さlenの16進文字列を数値にデコードしてresに格納
__host__ __device__ void small_hex_str_to_i64(const char *st, i32 len, i64 *res);

// 16進数のデコード
// 長さlenのstをsizeごとに切り分け、各部分文字列を16進数の数値と見て、サイズNのdataに数値として格納
__global__ void decode_hex_for_digits(const char *st, size_t len, i32 size, i64 *data, size_t N);

// Digitsのlook-ahead方式用の和
// a += b
__global__ void sum_for_look_ahead(i64 *a, const i64 *b, size_t len, i32 LOG_RADIX, char *ps, char *gs);

// Digitsのlook-ahead方式用の和、dataにcarryを足していく。最後の桁上げはcに格納
__global__ void sum_for_look_ahead_carry(i64 *data, char *carrys, size_t len, i32 LOG_RADIX, char *c);

// Digitsのlook-ahead方式用のcarryの計算
__global__ void carrys_for_look_ahead(char *ps, char *gs, size_t k, size_t len);

// Digitsの減算用のnot, (2^LOG_RADIX)-1の補数を求める
__global__ void not_for_complement(i64 *data, size_t len, i32 LOG_RADIX);
  
// 要素が最大値-1と等しいか
struct eq_radix {
    __host__ __device__
	bool operator()(i64 x) {
		return x == mmath::Digits::RADIX - 1;
	}
};
}

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

inline size_t mmath::Digits::first_non_radixmax_index() const {
    return thrust::distance(data.begin(), thrust::find_if_not(data.begin(), data.begin() + data.size(), Digits_utils::eq_radix()));
}

inline void mmath::Digits::to_zero() {
	thrust::device_vector<i64>().swap(data);
	data = thrust::device_vector<i64>(1, 0);
}

inline void mmath::Digits::push_msd(i64 num, size_t n) {
	for(size_t i = 0; i < n; i++) data.push_back(num);
}

inline void mmath::Digits::push_lsd(i64 num, size_t n) {
	data.insert(data.begin(), n, num);
}

inline void mmath::Digits::pop_msd() {
	data.pop_back();
}

inline void mmath::Digits::normalize() {
	while(size() > 1) {
		if(msd() == 0) pop_msd();
		else break;
	}
}

}
