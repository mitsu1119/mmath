#pragma once
#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include "util.cuh"
#include "ntt.cuh"

// add algorithm
// #define MMATH_DIGITS_ADD_SEQUENTIAL 1
#define MMATH_DIGITS_ADD_PARALLEL 1

// mul algorithm
// #define MMATH_DIGITS_MUL_SEQ_SCHOOL 1
#define MMATH_DIGITS_MUL_SEQ_NTT 1

namespace mmath {

using digit_type = u64;

class Digits {
private:
	// Digits(123456789) => data: {56789, 1234}
	thrust::device_vector<digit_type> data;

	// dataをlenの長さのvalで埋める
	Digits(size_t len, digit_type val);

public:
	Digits(digit_type val = 0);	

	// log2(digitsの基数)、つまり基数は2の何乗かを示す
	// 16進数への変換を考えて、LOG_RADIX % 4 = 0 である必要あり
	static constexpr i32 LOG_RADIX = 28;

	// 16進数として見たときの各桁の長さ
	static constexpr i32 LOG_16_RADIX = mmath::Digits::LOG_RADIX >> 2;

	// 基数
	static constexpr digit_type RADIX = (1 << mmath::Digits::LOG_RADIX);


	size_t size() const;
	digit_type at(size_t i) const;
	digit_type msd() const;

	// radix-1以外になる最初の要素番号
	size_t first_non_radixmax_index() const;

	void to_zero();		// dataを長さ1のゼロの配列にする
	void push_msd(digit_type num, size_t n = 1);
	void push_lsd(digit_type num, size_t n = 1);
	void pop_msd();

	void align();	// RADIXを超える桁を解決してきゃりーを伝播させる
	void normalize();	// 不必要な上位桁のゼロを削除する

	void add(const mmath::Digits &xx);	// this += x
	void sub(const mmath::Digits &x);	// this -= x, this >= x が前提
	void mul(const mmath::Digits &x);
	void increment();

	// 文字列を16進数と解釈して格納
	void from_hex(const char *st, size_t len_st);

	void print(bool hex = false) const;
};

namespace Digits_utils {
// 長さlenの16進文字列を数値にデコードしてresに格納
__host__ __device__ void small_hex_str_to_digit_type(const char *st, i32 len, digit_type *res);

// 16進数のデコード
// 長さlenのstをsizeごとに切り分け、各部分文字列を16進数の数値と見て、サイズNのdataに数値として格納
__global__ void decode_hex_for_digits(const char *st, size_t len, i32 size, digit_type *data, size_t N);

// Digitsのlook-ahead方式用の和
// a += b
__global__ void sum_for_look_ahead(digit_type *a, const digit_type *b, size_t len, i32 LOG_RADIX, char *ps, char *gs);

// Digitsのlook-ahead方式用の和、dataにcarryを足していく。最後の桁上げはcに格納
__global__ void sum_for_look_ahead_carry(digit_type *data, char *carrys, size_t len, i32 LOG_RADIX, char *c);

// Digitsのlook-ahead方式用のcarryの計算
__global__ void carrys_for_look_ahead(char *ps, char *gs, size_t k, size_t len);

// Digitsの減算用のnot, (2^LOG_RADIX)-1の補数を求める
__global__ void not_for_complement(digit_type *data, size_t len, i32 LOG_RADIX);
  
// 要素が最大値-1と等しいか
struct eq_radix {
    __host__ __device__
	bool operator()(digit_type x) {
		return x == mmath::Digits::RADIX - 1;
	}
};
}

inline mmath::Digits::Digits(size_t len, digit_type val): data(len, val) {
}

inline mmath::Digits::Digits(digit_type val): Digits(1, val) {
}

inline size_t mmath::Digits::size() const {
	return data.size();
}

inline digit_type mmath::Digits::at(size_t i) const {
	return data[i];
}

inline digit_type mmath::Digits::msd() const {
	return data.back();
}

inline size_t mmath::Digits::first_non_radixmax_index() const {
    return thrust::distance(data.begin(), thrust::find_if_not(data.begin(), data.begin() + data.size(), Digits_utils::eq_radix()));
}

inline void mmath::Digits::to_zero() {
	thrust::device_vector<digit_type>().swap(data);
	data = thrust::device_vector<digit_type>(1, 0);
}

inline void mmath::Digits::push_msd(digit_type num, size_t n) {
	for(size_t i = 0; i < n; i++) data.push_back(num);
}

inline void mmath::Digits::push_lsd(digit_type num, size_t n) {
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

inline void mmath::Digits::align() {
	for(size_t i = 0; i < size() - 1; i++) {
		if(data[i] >= RADIX) {
			data[i + 1] += data[i] / RADIX;
			data[i] %= RADIX;
		}
	}

	if(msd() >= RADIX) {
		push_msd(msd() / RADIX);
		data[size() - 2] %= RADIX;
	}
}

}
