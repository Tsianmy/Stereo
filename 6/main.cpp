#include <iostream>
#include <opencv2/opencv.hpp> 
using namespace std;
using namespace cv;

string path= "../left/", output_path = "../6output/";
const int xnum = 9, ynum = 6, imgnum = 14;
const float squareSize = 2.5e-02;

int main()
{
	vector<vector<Point2f>> imgPoints;
	Mat intrinsic_matrix, dist_coeffs;
	vector<float> reprojErrs;
	double totalAvgErr = 0;

	for (int id = 1; id <= imgnum; id++) {
		// read
		char buf[10];
		sprintf(buf, "%02d", id);
		string filename = path + "left" + buf + ".jpg";
		cout << "read " << filename << endl;

		Mat img = imread(filename), imgGray;
		if (img.empty()) {
			cout << "No such file." << endl;
			continue;
		}
		cvtColor(img, imgGray, COLOR_BGR2GRAY);

		// findChessboardCorners
		Size patternSize(xnum, ynum);
		vector<Point2f> corners;
		bool found = findChessboardCorners(imgGray, patternSize, corners, CALIB_CB_ADAPTIVE_THRESH);
		if (found) {
			cout << "Found corners." << endl;
			cornerSubPix(imgGray, corners, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
			imgPoints.push_back(corners);
			/*
			drawChessboardCorners(img, patternSize, corners, found);
			imshow("img", img);
			waitKey(0);
			*/
			vector<Mat> rvecs, tvecs;

			// calibration
			intrinsic_matrix = Mat::eye(3, 3, CV_64F);
			dist_coeffs = Mat::zeros(5, 1, CV_64F);
			vector<vector<Point3f>> objectPoints(1);
			for (int i = 0; i < xnum; i++)
				for (int j = 0; j < ynum; j++)
					objectPoints[0].push_back(Point3f(squareSize * i, squareSize * j, 0));
			objectPoints.resize(imgPoints.size(), objectPoints[0]);
			double err = calibrateCamera(objectPoints, imgPoints, img.size(), intrinsic_matrix,
				dist_coeffs, rvecs, tvecs, CALIB_ZERO_TANGENT_DIST | CALIB_FIX_PRINCIPAL_POINT);
			cout << "reprojection error: " << err << endl;

			if (id == imgnum) {
				int totalPoints = 0;
				vector<Point2f> imgPoints2;
				reprojErrs.resize(objectPoints.size());

				for (size_t i = 0; i < objectPoints.size(); i++)
				{
					projectPoints(Mat(objectPoints[i]), rvecs[i], tvecs[i],
						intrinsic_matrix, dist_coeffs, imgPoints2);
					double err = norm(Mat(imgPoints[i]), Mat(imgPoints2), NORM_L2);
					int sz = objectPoints[i].size();
					reprojErrs[i] = (float)sqrt(err * err / sz);
					totalAvgErr += err * err;
					totalPoints += sz;
				}
				totalAvgErr = sqrt(totalAvgErr / totalPoints);
			}
		}
	}
	FileStorage fs(output_path + "camera_data.xml", FileStorage::WRITE);
	fs << "intrinsic_matrix" << intrinsic_matrix
		<< "distortion_coefficients" << dist_coeffs
		<< "avg_reprojection_error" << totalAvgErr
		<< "per_view_reprojection_errors" << Mat(reprojErrs);
	fs.release();
	return 0;
}