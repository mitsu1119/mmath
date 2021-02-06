#include "Int.hpp"

mmath::Int::Int(): sign(PLUS) {
	digits.to_zero();
}

mmath::Int::Int(const Digits &digits): digits(digits), sign(PLUS) {
}


mmath::Int::Int(std::string_view x): sign(PLUS) {
	digits.to_zero();
	std::cout << "yey" << std::endl;
}
