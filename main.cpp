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
    std::string file_name = string(RES_DIR) + string("/3-2.jpeg");
    cv::Mat img = imread(file_name);
    cv::Mat ground_mask;

    SkyDetector sky_det;
    if (sky_det.extract_ground(img, ground_mask))
    {
        cout << "ground found" << endl;
        imshow("ground-mask", ground_mask);
        waitKey();
    }
    else
        cout << "no ground" << endl;

    destroyAllWindows();

    return 0;
}
