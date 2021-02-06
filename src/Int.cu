#include "Int.cuh"

mmath::Int::Int(): sign(PLUS) {
	digits.to_zero();
}

mmath::Int::Int(const Digits &digits): digits(digits), sign(PLUS) {
}


mmath::Int::Int(const std::string &x): sign(PLUS) {
	digits.from_hex(x.c_str(), x.size());
	digits.print(true);
}
