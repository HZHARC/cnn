#ifndef _CONVOLUTION_H
#define _CONVOLUTION_H
#include "timer.h"
#include "utils.h"
#include <vector>
#include <eigen3/Eigen/Dense>
<<<<<<< HEAD
#include <eigen3/Eigen/Core>


=======
>>>>>>> 5c36f3204a8cacbf4e2e19b3fd95fc5bc8f83f36

/* some constants definition */
#define IMG_HEIGHT 240
#define IMG_WIDTH 320
#define IMG_SIZE 76800

#define BLKNUM 1500   // should be able to read from inernal function directly

#define KERNEL_HEIGHT 7
#define KERNEL_WIDTH 7
#define KERNEL_SIZE 49



using namespace std;
<<<<<<< HEAD
using Eigen::MatrixXd;
using Eigen::Matrix;
=======
using MatrixXd;

>>>>>>> 5c36f3204a8cacbf4e2e19b3fd95fc5bc8f83f36



#endif
