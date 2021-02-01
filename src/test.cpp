#include <iostream>
#include "Int.hpp"

using namespace mmath;
int main() {
	Digits d;

	d.push_msd(100);
	d.push_msd(10);
	d.push_msd(0);
	d.push_msd(100);
	d.push_msd(0);
	d.push_msd(0);

	d.print();

	d.normalize();
	d.print();

	return 0;
}
