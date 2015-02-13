#include "convolution.h"
//Matrix<Double, Dynamic, Dynamic, RowMajor>
void batch(vector< vector < Matrix<Double, Dynamic, Dynamic, RowMajor> > > dataset, vector< Matrix<Double, Dynamic, Dynamic, RowMajor> > kernels, vector < vector < Matrix<Double, Dynamic, Dynamic, RowMajor> > > output );

int main(){
	
	Matrix<double, Dynamic, Dynamic, RowMajor> a = MatrixXd::Random(5,5);
	cout << a;
	

	return 0;
}
