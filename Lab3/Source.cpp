#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include "opencv2/calib3d.hpp"
#include <string>

using namespace std;
using namespace cv;

/*
@function hsv_conversion_on_one_channel
*/
static Mat hsv_conversion_on_one_channel(Mat src, int channel) {
	Mat hsv;

	cvtColor(src, hsv, COLOR_BGR2HSV);

	vector<Mat> channels;
	split(hsv, channels);
	equalizeHist(channels[channel], channels[channel]);

	Mat result;
	merge(channels, hsv);

	cvtColor(hsv, result, COLOR_HSV2BGR);

	return result;
}

/*
@function showHistogram
*/
// hists = vector of 3 cv::mat of size nbins=256 with the 3 histograms
// e.g.: hists[0] = cv:mat of size 256 with the red histogram
//       hists[1] = cv:mat of size 256 with the green histogram
//       hists[2] = cv:mat of size 256 with the blue histogram
void showHistogram(std::vector<cv::Mat>& hists)
{
	// Min/Max computation
	double hmax[3] = { 0,0,0 };
	double min;
	cv::minMaxLoc(hists[0], &min, &hmax[0]);
	cv::minMaxLoc(hists[1], &min, &hmax[1]);
	cv::minMaxLoc(hists[2], &min, &hmax[2]);

	std::string wname[3] = { "blue", "green", "red" };
	cv::Scalar colors[3] = { cv::Scalar(255,0,0), cv::Scalar(0,255,0),
							 cv::Scalar(0,0,255) };

	std::vector<cv::Mat> canvas(hists.size());

	// Display each histogram in a canvas
	for (int i = 0, end = hists.size(); i < end; i++)
	{
		canvas[i] = cv::Mat::ones(125, hists[0].rows, CV_8UC3);

		for (int j = 0, rows = canvas[i].rows; j < hists[0].rows - 1; j++)
		{
			cv::line(
				canvas[i],
				cv::Point(j, rows),
				cv::Point(j, rows - (hists[i].at<float>(j) * rows / hmax[i])),
				hists.size() == 1 ? cv::Scalar(200, 200, 200) : colors[i],
				1, 8, 0
			);
		}

		cv::imshow(hists.size() == 1 ? "value" : wname[i], canvas[i]);
	}
}

//+++++++++++++++++++++++++++++SECOND PART: FILTERING++++++++++++++++++++++++++++++++++++
//Global Variables
Mat img;
Mat dst;
Mat dst1;
Mat dst2;
const int kernel_slider_max = 20;
const int sigma_slider_max = 40;
const int sigma_space_max = 70;
const int sigma_color_max = 70;
int slider_value1;
int slider_value2;
int slider_value3;
int slider_value4;
int slider_value5;

struct ImageWithParams
{
	int param1;
	int param2;
	cv::Mat src_img, filtered_img;
};
/**
 * @function on_trackbar_mf
 * @brief Callback for trackbar in median filter
 */
void on_trackbar_mf(int, void* userdata)
{
	ImageWithParams* img = (ImageWithParams*)userdata;

	img->param1 = (double)2 * slider_value1 + 1;

	medianBlur(img->src_img, img->filtered_img, img->param1);
	imshow("Median filter", img->filtered_img);
}
/**
 * @function on_trackbar_gf
 * @brief Callback for trackbar in gaussian filter
 */
void on_trackbar_gf(int, void* userdata) {
	ImageWithParams* img = (ImageWithParams*)userdata;
	img->param1 = (double)2 * slider_value2 + 1;
	img->param2 = slider_value3;
	GaussianBlur(img->src_img, img->filtered_img, Size(img->param1, img->param1), img->param2, img->param2, BORDER_DEFAULT);
	imshow("Gaussian filter", img->filtered_img);
}
/**
 * @function on_trackbar_bf
 * @brief Callback for trackbar in bilateral filter
 */
void on_trackbar_bf(int, void* userdata) {
	ImageWithParams* img = (ImageWithParams*)userdata;
	img->param1 = slider_value4;
	img->param2 = slider_value5;
	bilateralFilter(img->src_img, img->filtered_img, 15, img->param1, img->param2, BORDER_DEFAULT);
	imshow("Bilateral filter", img->filtered_img);
}

int main() {

	Mat im = imread("countryside.jpg");
	if (!im.data) {
		cout << "Cannot read image." << endl;
		return -1;
	}

	imshow("Original image", im);

	// Separate the image in 3 places ( B, G and R )
	vector<Mat> bgr_planes;
	split(im, bgr_planes);

	// Establish the number of bins
	int histSize = 255;

	// Set the ranges ( for B,G,R) )
	float range[] = { 0, 255 };
	const float* histRange = { range };

	bool uniform = true; bool accumulate = false;

	Mat b_hist, g_hist, r_hist;

	// Compute the histograms:
	calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

	vector<Mat> histograms;
	histograms.push_back(b_hist);
	histograms.push_back(g_hist);
	histograms.push_back(r_hist);
	showHistogram(histograms);

	equalizeHist(bgr_planes[0], b_hist);
	equalizeHist(bgr_planes[1], g_hist);
	equalizeHist(bgr_planes[2], r_hist);

	//showing equalized image
	vector<Mat> combined;
	combined.push_back(b_hist);
	combined.push_back(g_hist);
	combined.push_back(r_hist);
	Mat result;
	merge(combined, result);

	cout << "Press any key on any showed window to see the new BGR image with related histograms. " << endl;

	waitKey(0);
	destroyAllWindows();

	imshow("BGR", result);

	vector<Mat> bgr_planes1;
	split(result, bgr_planes1);

	Mat b_hist1, g_hist1, r_hist1;

	// Compute the histograms:
	calcHist(&bgr_planes1[0], 1, 0, Mat(), b_hist1, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes1[1], 1, 0, Mat(), g_hist1, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes1[2], 1, 0, Mat(), r_hist1, 1, &histSize, &histRange, uniform, accumulate);

	vector<Mat> histograms1;
	histograms1.push_back(b_hist1);
	histograms1.push_back(g_hist1);
	histograms1.push_back(r_hist1);
	showHistogram(histograms1);

	cout << "Press any key on any showed window to see the new HSV image with related histograms. " << endl;

	waitKey(0);
	destroyAllWindows();

	Mat v_result = hsv_conversion_on_one_channel(im, 2);
	imshow("HSV image (equalized on V channel)", v_result);

	//showing histograms for the hsv image
	vector<Mat> hsv_planes;
	split(v_result, hsv_planes);
	Mat v_hist, h_hist, s_hist;

	// Compute the histograms:
	calcHist(&hsv_planes[0], 1, 0, Mat(), h_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&hsv_planes[1], 1, 0, Mat(), s_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&hsv_planes[2], 1, 0, Mat(), v_hist, 1, &histSize, &histRange, uniform, accumulate);

	vector<Mat> histogramshsv;
	histogramshsv.push_back(h_hist);
	histogramshsv.push_back(s_hist);
	histogramshsv.push_back(v_hist);
	showHistogram(histogramshsv);

	cout << "I guess that equalizing in the v channel gives a better result than the bgr one: colors are more vivid and clear. " << endl;
	cout << "Press any key on any showed window to go on with image filtering. " << endl;

	waitKey(0);
	destroyAllWindows();
	/*++++++++++++++++++++++++++++++++++++++++++IMAGE FILTERING++++++++++++++++++++++++++++++++++++++++*/

	// Read image and create clones for the other 2 filters
	img = imread("lena.png");
	Mat gaussian = img.clone();
	Mat bilateral = img.clone();

	if (!img.data) { printf("Error loading img \n"); return -1; };

	ImageWithParams median_filter{ slider_value1, -1, img.clone(), dst };
	// Create Windows
	namedWindow("Median filter", WINDOW_AUTOSIZE);
	//Create Trackbar
	createTrackbar("Kernel size", "Median filter", &slider_value1, kernel_slider_max, on_trackbar_mf, &median_filter);
	// trackbar on_change function
	on_trackbar_mf(slider_value1, &median_filter);

	//Gaussian filter
	ImageWithParams gaussian_filter{ slider_value2, slider_value3, gaussian, dst1 };
	namedWindow("Gaussian filter", WINDOW_AUTOSIZE);

	createTrackbar("Kernel size", "Gaussian filter", &slider_value2, kernel_slider_max, on_trackbar_gf, &gaussian_filter);
	createTrackbar("sigma size", "Gaussian filter", &slider_value3, sigma_slider_max, on_trackbar_gf, &gaussian_filter);
	on_trackbar_gf(slider_value2, &gaussian_filter);
	on_trackbar_gf(slider_value3, &gaussian_filter);

	//Bilateral filter
	ImageWithParams bilateral_filter{ slider_value4, slider_value5, bilateral, dst2 };
	namedWindow("Bilateral filter", WINDOW_AUTOSIZE);
	createTrackbar("sigma color", "Bilateral filter", &slider_value4, sigma_color_max, on_trackbar_bf, &bilateral_filter);
	createTrackbar("sigma space", "Bilateral filter", &slider_value5, sigma_space_max, on_trackbar_bf, &bilateral_filter);
	on_trackbar_bf(slider_value4, &bilateral_filter);
	on_trackbar_bf(slider_value5, &bilateral_filter);

	// Wait until user press some key
	waitKey(0);
	return 0;
}