#include <iostream>
#include <opencv2/opencv.hpp> 
using namespace std;
using namespace cv;

string path = "../left/", output_path = "../8output/";
const int xnum = 9, ynum = 6, imgnum = 14, distnum = 5;
const float squareSize = 2.5e-02;

void calcObjectPoints(vector<Point3f> & objectPoints)
{
	for (int j = 0; j < ynum; j++)
		for (int i = 0; i < xnum; i++)
			objectPoints.push_back(Point3f(i * squareSize, j * squareSize, 0));
}

void create_v(Mat & v, int i, int j, Mat & H)
{
	double arr[] = { H.at<double>(0, i) * H.at<double>(0, j),
		H.at<double>(0, i) * H.at<double>(1, j) + H.at<double>(1, i) * H.at<double>(0, j),
		H.at<double>(1, i) * H.at<double>(1, j),
		H.at<double>(0, i) * H.at<double>(2, j) + H.at<double>(2, i) * H.at<double>(0, j),
		H.at<double>(1, i) * H.at<double>(2, j) + H.at<double>(2, i) * H.at<double>(1, j),
		H.at<double>(2, i) * H.at<double>(2, j) };
	v = Mat(1, 6, CV_64F, arr).clone();
}

void get_intrinsics(Mat & intrinsic_matrix, vector<Mat> & Hs, int _imgnum)
{
	// [v_01; v_00 - v_01] b = 0
	Mat V;
	for (int i = 0; i < _imgnum; i++) {
		Mat v01, v00 , v11;
		create_v(v01, 0, 1, Hs[i]);
		create_v(v00, 0, 0, Hs[i]);
		create_v(v11, 1, 1, Hs[i]);
		V.push_back(v01);
		V.push_back(v00 - v11);
	}
	Mat mb = Mat::zeros(6, 1, CV_64F);
	SVD::solveZ(V, mb);
	Vec6f b = mb;
	float w = b[0] * b[2] * b[5] - b[1] * b[1] * b[5]
		- b[0] * b[4] * b[4] + 2 * b[1] * b[3] * b[4] - b[2] * b[3] * b[3];
	float d = b[0] * b[2] - b[1] * b[1];

	float alpha = sqrt(w / (d * b[0]));
	float beta = sqrt(w / (d * d) * b[0]);
	float gamma = sqrt(w / (d * d * b[0])) * b[1];
	float uc = (b[1] * b[4] - b[2] * b[3]) / d;
	float vc = (b[1] * b[3] - b[0] * b[4]) / d;
	intrinsic_matrix = (Mat_<double>(3, 3) << alpha, gamma, uc, 0, beta, vc, 0, 0, 1);
}

void get_extrinsics(vector<Mat> & rvecs, vector<Mat> & tvecs, Mat & intrinsic_matrix, vector<Mat> & Hs, int _imgnum)
{
	// A^-1
	Mat A_inv = intrinsic_matrix.inv();
	// [r1 r2 r3 t] = c[A^-1 * h1, A^-1 * h2, r1 x r2, A^-1 * h3]
	for (int i = 0; i < _imgnum; i++) {
		Mat R, h[3] = { Hs[i].col(0), Hs[i].col(1),  Hs[i].col(2) };
		Mat rt[3] = { A_inv * h[0] , A_inv * h[1], A_inv * h[2] };
		for (int i = 0; i < 2; i++) {
			rt[i] /= norm(rt[i]);
			R.push_back(rt[i].t());
		}
		rt[2] /= norm(A_inv * h[0]);
		R.push_back(rt[0].cross(rt[1]).t());
		rvecs.push_back(R.t());
		tvecs.push_back(rt[2]);
	}
}

void get_distortion(vector<vector<Point3f>> & objectPoints, vector<vector<Point2f>> & imgPoints,
	Mat & intrinsic_matrix, Mat & dist_coeffs, vector<Mat> & rvecs, vector<Mat> & tvecs,
	int _imgnum)
{
	// D k = d
	Mat D, d;
	double u0 = intrinsic_matrix.at<double>(0, 2),
		v0 = intrinsic_matrix.at<double>(1, 2);
	for (int i = 0; i < _imgnum; i++) {
		for (int j = 0; j < imgPoints[i].size(); j++) {
			Mat ObjX = (Mat_<double>(3, 1) << objectPoints[i][j].x, objectPoints[i][j].y, objectPoints[i][j].z);
			
			vector<double> X = vector<double>(Mat(intrinsic_matrix * (rvecs[i] * ObjX + tvecs[i])));
			double u = X[0] / X[2], v = X[1] / X[2];
			
			vector<double> X_c = vector<double>(Mat(rvecs[i] * ObjX + tvecs[i]));
			X_c[0] /= X_c[2]; X_c[1] /= X_c[2];
			double r2 = X_c[0] * X_c[0] + X_c[1] * X_c[1] + 1;
			Mat rr = (Mat_<double>(1, 2) << r2, r2 * r2);

			// [(u - u_0)r^2, (u - u_0)r^4]
			// [(v - v_0)r^2, (v - v_0)r^4]
			D.push_back((u - u0) * rr);
			D.push_back((v - v0) * rr);
			// [u_d - u]
			// [v_d - v]
			d.push_back(imgPoints[i][j].x - u);
			d.push_back(imgPoints[i][j].y - v);
		}
	}
	Mat tmp = (D.t() * D).inv() * D.t() * d;
	dist_coeffs.release();
	dist_coeffs = Mat::zeros(distnum, 1, CV_64F);
	tmp.copyTo(dist_coeffs.rowRange(0, 2));
}

double runCalibration(vector<vector<Point3f>> & objectPoints, vector<vector<Point2f>> & imgPoints,
	Size imgSize, Mat & intrinsic_matrix, Mat & dist_coeffs,
	vector<Mat> & rvecs, vector<Mat> & tvecs)
{
	int _imgnum = objectPoints.size();
	vector<Mat> Hs;
	for (int i = 0; i < _imgnum; i++) {
		Hs.emplace_back(findHomography(objectPoints[i], imgPoints[i], noArray()));
	}
	get_intrinsics(intrinsic_matrix, Hs, _imgnum);
	cout << "intrinsics\n" << intrinsic_matrix << endl;
	get_extrinsics(rvecs, tvecs, intrinsic_matrix, Hs, _imgnum);
	get_distortion(objectPoints, imgPoints, intrinsic_matrix, dist_coeffs, rvecs, tvecs, _imgnum);
	cout << "dist_coeffs\n" << dist_coeffs << endl;

	// error
	vector<float> reprojErrs(objectPoints.size());
	vector<Point2f> imgPoints2;
	double totalAvgErr = 0;
	int totalPoints = 0;
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
	return totalAvgErr;
}


int main()
{
	vector<vector<Point2f>> imgPoints;
	Size imgSize;

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
		imgSize = img.size();
		cvtColor(img, imgGray, COLOR_BGR2GRAY);

		// findChessboardCorners
		Size patternSize(xnum, ynum);
		vector<Point2f> corners;
		bool found = findChessboardCorners(imgGray, patternSize, corners);
		if (found) {
			cout << "Found corners." << endl;
			cornerSubPix(imgGray, corners, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1));
			imgPoints.push_back(corners);
		}
	}

	// calibration
	Mat intrinsic_matrix , dist_coeffs;
	vector<Mat> rvecs, tvecs;
	vector<vector<Point3f>> objectPoints(1);
	calcObjectPoints(objectPoints[0]);
	objectPoints.resize(imgPoints.size(), objectPoints[0]);
	double err = runCalibration(objectPoints, imgPoints, imgSize, intrinsic_matrix, dist_coeffs,
		rvecs, tvecs);
	cout << "reprojection error: " << err << endl;


	FileStorage fs(output_path + "camera_data.xml", FileStorage::WRITE);
	fs << "image_size" << imgSize
		<< "intrinsic_matrix" << intrinsic_matrix
		<< "distortion_coefficients" << dist_coeffs
		<< "avg_reprojection_error" << err;
	fs.release();

	// undistort
	Mat map1, map2;
	initUndistortRectifyMap(intrinsic_matrix, dist_coeffs, Mat(),
		getOptimalNewCameraMatrix(intrinsic_matrix, dist_coeffs, imgSize, 1, imgSize, 0),
		imgSize, CV_16SC2, map1, map2);
	for (int id = 1; id <= imgnum; id++) {
		// read
		char buf[10];
		sprintf(buf, "%02d", id);
		string filename = path + "left" + buf + ".jpg";
		Mat img = imread(filename);
		if (img.empty()) continue;

		Mat rimg;
		remap(img, rimg, map1, map2, INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar());
		string outname = output_path + "rleft" + buf + ".jpg";
		cout << "save  " << filename << endl;
		imwrite(outname, rimg);
	}
	return 0;
}