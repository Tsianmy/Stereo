#include <iostream>
#include <opencv2/opencv.hpp> 
using namespace std;
using namespace cv;

string path = "../left/", input_path = "../6output/", output_path = "../7output/";
const int xnum = 9, ynum = 6, imgnum = 14;
const float squareSize = 50.0;

int main()
{
	FileStorage fs(input_path + "camera_data.xml", FileStorage::READ);
	Size imgSize;
	cv::Mat intrinsic_matrix, dist_coeffs;
	fs["image_size"] >> imgSize;
	fs["intrinsic_matrix"] >> intrinsic_matrix;
	fs["distortion_coefficients"] >> dist_coeffs;
	cout << "image size: " << imgSize << endl
		 << "intrinsic matrix:\n" << intrinsic_matrix << endl
		 << "distortion coefficients:\n" << dist_coeffs << endl;
	Mat map1, map2;
	initUndistortRectifyMap(intrinsic_matrix, dist_coeffs, Mat(),
		intrinsic_matrix,
		imgSize, CV_16SC2, map1, map2);
	for (int id = 1; id <= imgnum; id++) {
		// read
		char buf[10];
		sprintf(buf, "%02d", id);
		string filename = path + "left" + buf + ".jpg";
		cout << "read " << filename << endl;

		Mat img = imread(filename);
		if (img.empty()) {
			cout << "No such file." << endl;
			continue;
		}
		Mat rimg;
		remap(img, rimg, map1, map2, INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar());
		//undistort(img, rimg, intrinsic_matrix, dist_coeffs);
		imwrite(output_path + "rleft" + buf + ".jpg", rimg);
	}
	return 0;
}