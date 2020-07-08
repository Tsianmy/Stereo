#include <iostream>
#include <opencv2/opencv.hpp> 
using namespace std;
using namespace cv;

string path[] = { "../left/" , "../right/" }, output_path = "../12output/";
const int xnum = 9, ynum = 6, imgnum = 14;
Size imgSize;
const float squareSize = 1;

void runCalibration(string ori, vector<vector<Point3f>> & objectPoints,
	vector<vector<Point2f>> & imgPoints, Mat & intrinsic_matrix, Mat & dist_coeffs,
	vector<Mat> & rvecs, vector<Mat> & tvecs)
{
	// calibration
	cout << "calibrate " << ori << endl;
	intrinsic_matrix = Mat::eye(3, 3, CV_64F);
	dist_coeffs = Mat::zeros(5, 1, CV_64F);

	double err = calibrateCamera(objectPoints, imgPoints, imgSize, intrinsic_matrix,
		dist_coeffs, rvecs, tvecs);
	cout << "intrinsics: \n" << intrinsic_matrix << endl
		 << "distortion coefficients: \n" << dist_coeffs << endl
		 << "reprojection error: " << err << endl;

	int MODE = ori == "right" ? FileStorage::APPEND : FileStorage::WRITE;
	FileStorage fs(output_path + "camera_data.xml", MODE);
	fs << ori << "[";
	fs << "image_size" << imgSize
		<< "intrinsic_matrix" << intrinsic_matrix
		<< "distortion_coefficients" << dist_coeffs
		<< "avg_reprojection_error" << err;
	fs << "]";
	fs.release();
}

int main()
{
	vector<vector<Point2f>> imgPoints[2];
	Size boardSize(xnum, ynum);

	cout << "read" << endl;
	string ori[] = { "left", "right" };
	const int maxScale = 2;
	for (int id = 1; id <= imgnum; id++) {
		char buf[10];
		sprintf(buf, "%02d", id);
		vector<Point2f> corners[2];
		int k;
		for (k = 0; k < 2; k++) {
			string filename = path[k] + ori[k] + buf + ".jpg";
			Mat img = imread(filename), imgGray;
			if (img.empty()) break;
			imgSize = img.size();

			// findChessboardCorners
			cvtColor(img, imgGray, COLOR_BGR2GRAY);
			bool found = false;
			found = findChessboardCorners(imgGray, boardSize, corners[k]);
			/*for (int scale = 1; scale <= maxScale; scale++) {
				Mat tempImg;
				if (scale == 1) tempImg = imgGray;
				else resize(imgGray, tempImg, Size(), scale, scale, INTER_LINEAR_EXACT);
				found = findChessboardCorners(tempImg, boardSize, corners[k], CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
				if (found) {
					if (scale > 1) {
						Mat cornersMat(corners[k]);
						cornersMat *= 1. / scale;
					}
					break;
				}
			}*/
			if (!found) break;
			cornerSubPix(imgGray, corners[k], Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1));
		}
		if (k == 2) {
			cout << buf << ": found corners." << endl;
			imgPoints[0].push_back(corners[0]);
			imgPoints[1].push_back(corners[1]);
		}
	}

	Mat intrinsic_matrix[2], dist_coeffs[2];
	vector<Mat> rvecsL, rvecsR, tvecsL, tvecsR;
	vector<vector<Point3f>> objectPoints(1);
	for (int j = 0; j < ynum; j++)
		for (int i = 0; i < xnum; i++)
			objectPoints[0].push_back(Point3f(i * squareSize, j * squareSize, 0));
	objectPoints.resize(imgPoints[0].size(), objectPoints[0]);
	int nimages = objectPoints.size();

	/*Mat cameraMatrix[2], distCoeffs[2];
	cameraMatrix[0] = initCameraMatrix2D(objectPoints, imgPoints[0], imgSize, 0);
	cameraMatrix[1] = initCameraMatrix2D(objectPoints, imgPoints[1], imgSize, 0);
	Mat R, T, E, F;

	double rms = stereoCalibrate(objectPoints, imgPoints[0], imgPoints[1],
		cameraMatrix[0], distCoeffs[0],
		cameraMatrix[1], distCoeffs[1],
		imgSize, R, T, E, F,
		CALIB_FIX_ASPECT_RATIO +
		CALIB_ZERO_TANGENT_DIST +
		CALIB_USE_INTRINSIC_GUESS +
		CALIB_SAME_FOCAL_LENGTH +
		CALIB_RATIONAL_MODEL +
		CALIB_FIX_K3 + CALIB_FIX_K4 + CALIB_FIX_K5,
		TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, 1e-5));
	cout << "done with RMS error=" << rms << endl;
	cout << "R:\n" << R << endl;*/

	runCalibration("left", objectPoints, imgPoints[0], intrinsic_matrix[0], dist_coeffs[0], rvecsL, tvecsL);
	runCalibration("right", objectPoints, imgPoints[1], intrinsic_matrix[1], dist_coeffs[1], rvecsR, tvecsR);

	vector<Point3f> leftPoints;
	vector<Point2f> _imgPoints;
	cout << rvecsL[0].size() << endl;
	for (int i = 0; i < nimages; i++) {
		for (int j = 0; j < objectPoints[i].size(); j++) {
			Mat pl = (Mat_<double>(3, 1) << objectPoints[i][j].x, objectPoints[i][j].y, objectPoints[i][j].z);
			Mat R;
			Rodrigues(rvecsL[i], R);
			pl = R * pl + tvecsL[i];
			vector<double> vec = vector<double>(pl);
			leftPoints.push_back(Point3f(vec[0], vec[1], vec[2]));
		}
		_imgPoints.insert(_imgPoints.end(), imgPoints[1][i].begin(), imgPoints[1][i].end());
	}
	cout << _imgPoints.size() << endl;
	Mat rodR, R, T;
	solvePnP(leftPoints, _imgPoints, intrinsic_matrix[1], dist_coeffs[1], rodR, T);
	Rodrigues(rodR, R);
	cout << R << endl << T << endl;

	// error
	double err = 0;
	int npoints = 0;
	vector<Vec3f> lines[2];
	for (int i = 0; i < nimages; i++)
	{
		int npt = (int)imgPoints[0][i].size();
		Mat imgpt[2];
		for (int k = 0; k < 2; k++)
		{
			imgpt[k] = Mat(imgPoints[k][i]);
			undistortPoints(imgpt[k], imgpt[k], intrinsic_matrix[k], dist_coeffs[k], Mat(), intrinsic_matrix[k]);
			computeCorrespondEpilines(imgpt[k], k + 1, F, lines[k]);
		}
		for (int j = 0; j < npt; j++)
		{
			double errij = fabs(imgPoints[0][i][j].x*lines[1][j][0] +
				imgPoints[0][i][j].y*lines[1][j][1] + lines[1][j][2]) +
				fabs(imgPoints[1][i][j].x*lines[0][j][0] +
					imgPoints[1][i][j].y*lines[0][j][1] + lines[0][j][2]);
			err += errij;
		}
		npoints += npt;
	}
	cout << "average epipolar err = " << err / npoints << endl;
	return 0;
}