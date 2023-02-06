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
    /*
    int l = 3;
    for (int i=1; i<=6; ++i)
    {
        std::string name = to_string(l) + "-" + to_string(i) + ".jpeg";
        //std::string name = to_string(i) + ".jpeg";
        std::string file_name = string(DATA_DIR) + "/" + name;
        std::string res_name = string(RES_DIR) + "/mask-" + name;
        cv::Mat img = imread(file_name);

        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        double s = cv::sum(gray)[0];
        double N = gray.rows * gray.cols;
        cout << "img height: " << img.rows << endl;
        cout << "img width: " << img.cols << endl;
        cout << "Mean Brightness:  " << s/N << endl;

        SkyDetector sky_det;
        cv::Mat ground_mask;
        if (sky_det.extract_ground(img, ground_mask))
        {
            cout << "ground found" << endl;
            double pixels_num = cv::countNonZero(ground_mask);
            double N = ground_mask.rows * ground_mask.cols;
            double ratio = pixels_num / N;
            cout << "ground ratio: " << ratio << endl;
            imwrite(res_name, ground_mask);
        }
        else
            cout << "ground NOT found" << endl;

        cout << to_string(i) << "_th image finished" << endl << endl;
    }
    */


    
    //std::string file_name = string(RES_DIR) + "/" + to_string(i) + ".jpeg";
    std::string file_ext = ".jpeg";
    std::string name;
    std::cout << "Enter file name without extension: ";
    std::cin >> name;
    name += file_ext;
    
    std::string file_name = string(DATA_DIR) + "/" + name;
    std::string res_name = string(RES_DIR) + "/mask-" + name;
    cv::Mat img = imread(file_name);
    cv::Mat ground_mask;

    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    double s = cv::sum(gray)[0];
    double N  = gray.rows * gray.cols;
    cout << "img height: " << img.rows << endl;
    cout << "img width: " << img.cols << endl;
    cout << "Mean Brightness:  " << s/N << endl;

    SkyDetector sky_det;
    if (sky_det.extract_ground(img, ground_mask))
    {
        cout << "ground found" << endl;
        double pixels_num = cv::countNonZero(ground_mask);
        double N = ground_mask.rows * ground_mask.cols;
        double ratio = pixels_num / N;
        cout << "ground ratio: " << ratio << endl;
        imwrite(res_name, ground_mask);
    }
    else
        cout << "ground NOT found" << endl;
    
    return 0;
}
