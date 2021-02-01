#pragma once
#include <vector>
#include "util.hpp"

namespace mmath {

class Digits {
private:
	// Digits(12345678) => data: {78, 56, 34, 12}
	std::vector<i64> data;
	
	// dataをlenの長さのvalで埋める
	Digits(size_t len, i64 val);

public:
	Digits(i64 val = 0);

	size_t size() const;
	size_t msd() const;

	void to_zero();		// dataを長さ1のゼロの配列にする
	void push_msd(i64 num);
	void push_msd_zero(size_t n);
	void pop_msd();
	void resize(size_t size);
	void reserve(size_t size);
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
	std::vector<i64>().swap(data);
	data = {0};
}

inline void mmath::Digits::push_msd(i64 num) {
	data.push_back(num);
}

inline void mmath::Digits::push_msd_zero(size_t n) {
	data.insert(data.begin(), n, 0);
}

inline void mmath::Digits::pop_msd() {
	data.pop_back();
}

inline void mmath::Digits::resize(size_t size) {
	data.resize(size, 0);
}

inline void mmath::Digits::reserve(size_t size) {
	data.reserve(size);
}

inline void mmath::Digits::normalize() {
	while(size() > 0) {
		if(msd() == 0) pop_msd();
		else break;
	}

	if(size() == 0) to_zero();
}

void mmath::Digits::print() const {
	for(auto &i: data) std::cout << i << " ";
	std::cout << std::endl;
}

}
