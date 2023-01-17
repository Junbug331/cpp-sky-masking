#include <iostream>
#include <algorithm>
#include <vector>
#include <numeric>
#include <string>

#include "sky_detector.hpp"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main()
{
    std::string file_name = string(RES_DIR) + string("/5.jpeg");
    cv::Mat img = imread(file_name);
    cv::Mat sky_mask, ground_mask;

    SkyDetector sky_det;
    if (sky_det.extract_sky(img, sky_mask))
    {
        cout << "sky found" << endl;
        imshow("sky-mask", sky_mask);
    }
    else
        cout << "no sky" << endl;

    if (sky_det.extract_sky(img, ground_mask))
    {
        cout << "ground found" << endl;
        imshow("ground-mask", ground_mask);
    }
    else
        cout << "no sky" << endl;
    cout << "finished" << endl;

    return 0;
}
