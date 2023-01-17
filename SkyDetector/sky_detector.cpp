#include <vector>
#include <cassert>
#include <numeric>
#include <algorithm>

#include "sky_detector.hpp"

void SkyDetector::extract_image_gradient(const cv::Mat &img, cv::Mat &gradient_img)
{
    cv::Mat gray_img;
    cv::cvtColor(img, gray_img, cv::COLOR_BGR2GRAY);

    // Sobel operator to extract image gradient
    cv::Mat x_gradient, y_gradient;
    cv::Sobel(gray_img, x_gradient, CV_64F, 1, 0);
    cv::Sobel(gray_img, y_gradient, CV_64F, 0, 1);

    // Calculate the gradient information map
    cv::Mat gradient;
    cv::pow(x_gradient, 2, x_gradient);
    cv::pow(y_gradient, 2, y_gradient);
    cv::add(x_gradient, y_gradient, gradient);
    cv::sqrt(gradient, gradient);

    gradient_img = gradient;
}

void SkyDetector::extract_border(const cv::Mat &gradient_info_map, double thresh, std::vector<int> &border)
{
    int img_H = gradient_info_map.rows;
    int img_W = gradient_info_map.cols;
    border = std::vector<int>(img_W, img_H - 1);

    for (int col = 0; col < img_W; ++col)
    {
        int row_idx = 0;
        for (int row = 0; row < img_H; ++row)
        {
            row_idx = row;
            if (gradient_info_map.at<double>(row, col) > thresh)
            {
                border.at(col) = row;
                break;
            }
        }
        if (row_idx == 0) // no sky
        {
            border.at(col) = img_H - 1;
        }
    }
}

cv::Mat SkyDetector::make_sky_mask(const cv::Mat &img, const std::vector<int> &border, int type)
{
    int img_H = img.rows;
    int img_W = img.cols;

    cv::Mat mask = cv::Mat::zeros(img_H, img_W, CV_8UC1);

    if (type == 1) // sky mask sky: 255, ground: 0
    {
        for (int col = 0; col < img_W; ++col)
        {
            for (int row = 0; row < img_H; ++row)
            {
                if (row <= border.at(col))
                    mask.at<uchar>(row, col) = 255;
            }
        }
    }
    else if (type == 0) // ground mask sky: 0, ground: 255
    {
        for (int col = 0; col < img_W; ++col)
        {
            for (int row = 0; row < img_H; ++row)
            {
                if (row > border.at(col))
                    mask.at<uchar>(row, col) = 255;
            }
        }
    }
    else
        assert(type == 0 || type == 1);

    return mask;
}

double SkyDetector::calculate_sky_energy(const std::vector<int> &border, const cv::Mat &img)
{
    int img_H = img.rows;
    int img_W = img.cols;

    // make sky image mask and ground image mask
    cv::Mat sky_mask = make_sky_mask(img, border, 1);
    cv::Mat ground_mask = make_sky_mask(img, border, 0);

    // Deduct sky image and ground image
    cv::Mat sky_img = cv::Mat::zeros(img_H, img_W, CV_8UC3);
    cv::Mat ground_img = cv::Mat::zeros(img_H, img_W, CV_8UC3);
    img.copyTo(sky_img, sky_mask);
    img.copyTo(ground_img, ground_mask);

    // Calculate the sky and ground image cov matrix
    int ground_non_zeros_nums = cv::countNonZero(ground_mask);
    int sky_non_zeros_nums = cv::countNonZero(sky_mask);

    if (ground_non_zeros_nums == 0 || sky_non_zeros_nums == 0)
        return std::numeric_limits<double>::min();

    cv::Mat ground_img_non_zero = cv::Mat::zeros(ground_non_zeros_nums, 3, CV_8UC1);
    cv::Mat sky_img_non_zero = cv::Mat::zeros(sky_non_zeros_nums, 3, CV_8UC1);

    int row_index = 0;
    for (int col = 0; col < ground_img.cols; ++col)
    {
        for (int row = 0; row < ground_img.rows; ++row)
        {
            if (ground_img.at<cv::Vec3b>(row, col)[0] == 0 &&
                ground_img.at<cv::Vec3b>(row, col)[1] == 0 &&
                ground_img.at<cv::Vec3b>(row, col)[2] == 0)
            {
                continue;
            }
            else
            {
                cv::Vec3b intensity = ground_img.at<cv::Vec3b>(row, col);
                ground_img_non_zero.at<uchar>(row_index, 0) = intensity[0];
                ground_img_non_zero.at<uchar>(row_index, 1) = intensity[1];
                ground_img_non_zero.at<uchar>(row_index, 2) = intensity[2];
                row_index++;
            }
        }
    }

    row_index = 0;
    for (int col = 0; col < sky_img.cols; ++col)
    {
        for (int row = 0; row < sky_img.rows; ++row)
        {
            if (sky_img.at<cv::Vec3b>(row, col)[0] == 0 &&
                sky_img.at<cv::Vec3b>(row, col)[1] == 0 &&
                sky_img.at<cv::Vec3b>(row, col)[2] == 0)
            {
                continue;
            }
            else
            {
                cv::Vec3b intensity = sky_img.at<cv::Vec3b>(row, col);
                sky_img_non_zero.at<uchar>(row_index, 0) = intensity[0];
                sky_img_non_zero.at<uchar>(row_index, 1) = intensity[1];
                sky_img_non_zero.at<uchar>(row_index, 2) = intensity[2];
                row_index++;
            }
        }
    }

    /*
     - ground_mean : 3x1 column vector that represent average RGB values
     - cv::COVAR_SCALE : "normal" mode, scale is 1./nsamples, not 1./(nsamples-1)
    */
    cv::Mat ground_covar, ground_mean, ground_eig_vec, ground_eig_val;
    cv::calcCovarMatrix(ground_img_non_zero, ground_covar, ground_mean,
                        cv::COVAR_ROWS | cv::COVAR_NORMAL | cv::COVAR_SCALE);
    cv::eigen(ground_covar, ground_eig_val, ground_eig_vec);

    cv::Mat sky_covar, sky_mean, sky_eig_vec, sky_eig_val;
    cv::calcCovarMatrix(sky_img_non_zero, sky_covar, sky_mean,
                        cv::COVAR_ROWS | cv::COVAR_SCALE | cv::COVAR_NORMAL);
    cv::eigen(sky_covar, sky_eig_val, sky_eig_vec);

    int para = 2; // the original parameters of the thesis
    double ground_det = cv::determinant(ground_covar);
    double sky_det = cv::determinant(sky_covar);
    double ground_eig_det = cv::determinant(ground_eig_vec);
    double sky_eig_det = cv::determinant(sky_eig_vec);

    // Ideal sky region would have low variance
    // Note that maximizing 1/(...) would minimize the variance of the ground and sky distributions
    return 1 / ((para * sky_det + ground_det) + (para * sky_eig_det + ground_eig_det));
}

void SkyDetector::extract_border_optimal(const cv::Mat &img, std::vector<int> &sky_border_optimal)
{
    // Extract the gradient information map
    cv::Mat gradient_info_map;
    extract_image_gradient(img, gradient_info_map);

    // max = 600, min = 5, step = 5, n = 119
    int n = static_cast<int>(std::floor((f_thres_sky_max - f_thres_sky_min) / f_thres_sky_search_step)) + 1;

    double jn_max = 0.0;

    for (int k = 1; k < n + 1; ++k)
    {
        // t : threshold for gradient value t : f_thres_sky_min ~ f_thres_sky_max
        double t = f_thres_sky_min + (std::floor((f_thres_sky_max - f_thres_sky_min) / n) - 1) * (k - 1);

        // b_tmp(col) : sky boundary value of row
        // 0 ~ b(col) : sky
        // b(col) ~ H : ground
        std::vector<int> b_tmp;
        extract_border(gradient_info_map, t, b_tmp);
        double jn = calculate_sky_energy(b_tmp, img);
        if (std::isinf(jn))
        {
            // jn is numberic_limits<double>::min();
            std::cout << "Jn is -inf" << std::endl;
        }
        if (jn > jn_max)
        {
            jn_max = jn;
            sky_border_optimal = b_tmp;
        }
    }
}

/***
 * Determine if the image contains a sky region
 * @param border
 * @param thresh_1
 * @param thresh_2
 * @param thresh_3
 * @return
 */
bool SkyDetector::has_sky_region(const std::vector<int> &border,
                                 double thresh_1, double thresh_2, double thresh_3)
{
    double border_mean = 0.0;
    border_mean = std::accumulate(border.begin(), border.end(), 0.0);
    border_mean /= static_cast<double>(border.size());

    // If the avg skyline is too small considr no sky area
    if (border_mean < thresh_1)
        return false;

    std::vector<int> border_diff(border.size());
    std::adjacent_difference(border.begin(), border.end(), border_diff.begin(),
                             [](const int &a, const int &b)
                             { return std::abs(a - b); });
    double border_diff_mean = 0.0; // ASADSBP
    border_diff_mean = std::accumulate(border_diff.begin() + 1, border_diff.end(), 0.0);
    border_diff_mean /= static_cast<double>(border.size());
    // Large ASADSBP means frequent chagnes in the sky border position function

    return !(border_mean < thresh_1 || (border_diff_mean > thresh_3 && border_mean < thresh_2));
}

bool SkyDetector::has_partial_sky_region(const std::vector<int> &border, double thresh_1)
{
    std::vector<int> border_diff(border.size());
    std::adjacent_difference(border.begin(), border.end(), border_diff.begin(),
                             [](const int &a, const int &b)
                             { return std::abs(a - b); });
    for (auto it = border_diff.begin() + 1; it != border_diff.end(); ++it)
    {
        if (*it > thresh_1)
            return true;
    }

    return false;
}

void SkyDetector::refine_border(const std::vector<int> &border, const cv::Mat &img, std::vector<int> &refined_border)
{
    int image_height = img.size[0];
    int image_width = img.size[1];

    // make sky image mask and ground image mask
    cv::Mat sky_mask = make_sky_mask(img, border, 1);
    cv::Mat ground_mask = make_sky_mask(img, border, 0);

    // Deduct sky image and ground image
    cv::Mat sky_image = cv::Mat::zeros(image_height, image_width, CV_8UC3);
    cv::Mat ground_image = cv::Mat::zeros(image_height, image_width, CV_8UC3);
    img.copyTo(sky_image, sky_mask);
    img.copyTo(ground_image, ground_mask);

    // Calculate the sky and ground image covariance matrix
    int ground_non_zeros_nums = cv::countNonZero(ground_mask);
    int sky_non_zeros_nums = cv::countNonZero(sky_mask);

    cv::Mat ground_image_non_zero = cv::Mat::zeros(ground_non_zeros_nums, 3, CV_8UC1);
    cv::Mat sky_image_non_zero = cv::Mat::zeros(sky_non_zeros_nums, 3, CV_8UC1);

    int row_index = 0;
    for (int col = 0; col < ground_image.cols; ++col)
    {
        for (int row = 0; row < ground_image.rows; ++row)
        {
            if (ground_image.at<cv::Vec3b>(row, col)[0] == 0 &&
                ground_image.at<cv::Vec3b>(row, col)[1] == 0 &&
                ground_image.at<cv::Vec3b>(row, col)[2] == 0)
            {
                continue;
            }
            else
            {
                cv::Vec3b intensity = ground_image.at<cv::Vec3b>(row, col);
                ground_image_non_zero.at<uchar>(row_index, 0) = intensity[0];
                ground_image_non_zero.at<uchar>(row_index, 1) = intensity[1];
                ground_image_non_zero.at<uchar>(row_index, 2) = intensity[2];
                row_index++;
            }
        }
    }

    row_index = 0;
    for (int col = 0; col < sky_image.cols; ++col)
    {
        for (int row = 0; row < sky_image.rows; ++row)
        {
            if (sky_image.at<cv::Vec3b>(row, col)[0] == 0 &&
                sky_image.at<cv::Vec3b>(row, col)[1] == 0 &&
                sky_image.at<cv::Vec3b>(row, col)[2] == 0)
            {
                continue;
            }
            else
            {
                cv::Vec3b intensity = sky_image.at<cv::Vec3b>(row, col);
                sky_image_non_zero.at<uchar>(row_index, 0) = intensity[0];
                sky_image_non_zero.at<uchar>(row_index, 1) = intensity[1];
                sky_image_non_zero.at<uchar>(row_index, 2) = intensity[2];
                row_index++;
            }
        }
    }

    // Applying K-means Algorithm to separate the sky region into clusters.
    // u_s1, Sigma_s1, u_s2, Sigma_s2
    // labels {0, 1} N_sample x 1
    cv::Mat sky_image_float;
    sky_image_non_zero.convertTo(sky_image_float, CV_32FC1);
    cv::Mat labels;
    cv::kmeans(sky_image_float, 2, labels,
               cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 10, 1.0), 10, cv::KMEANS_RANDOM_CENTERS);
    int label_1_nums = cv::countNonZero(labels);
    int label_0_nums = labels.rows - label_1_nums;

    cv::Mat sky_label_1_image = cv::Mat::zeros(label_1_nums, 3, CV_8UC1);
    cv::Mat sky_label_0_image = cv::Mat::zeros(label_0_nums, 3, CV_8UC1);

    row_index = 0;
    for (int row = 0; row < labels.rows; ++row)
    {
        if (labels.at<float>(row, 0) == 0.0)
        {
            sky_label_0_image.at<uchar>(row_index, 0) = sky_image_non_zero.at<uchar>(row, 0);
            sky_label_0_image.at<uchar>(row_index, 1) = sky_image_non_zero.at<uchar>(row, 1);
            sky_label_0_image.at<uchar>(row_index, 2) = sky_image_non_zero.at<uchar>(row, 2);
            row_index++;
        }
    }
    row_index = 0;
    for (int row = 0; row < labels.rows; ++row)
    {
        if (labels.at<float>(row, 0) == 1.0)
        {
            sky_label_1_image.at<uchar>(row_index, 0) = sky_image_non_zero.at<uchar>(row, 0);
            sky_label_1_image.at<uchar>(row_index, 1) = sky_image_non_zero.at<uchar>(row, 1);
            sky_label_1_image.at<uchar>(row_index, 2) = sky_image_non_zero.at<uchar>(row, 2);
            row_index++;
        }
    }

    cv::Mat sky_covar_1, sky_mean_1;
    cv::calcCovarMatrix(sky_label_1_image, sky_covar_1, sky_mean_1,
                        cv::COVAR_ROWS | cv::COVAR_NORMAL | cv::COVAR_SCALE);
    cv::Mat ic_s1;
    cv::invert(sky_covar_1, ic_s1, cv::DECOMP_SVD);

    cv::Mat sky_covar_0, sky_mean_0;
    cv::calcCovarMatrix(sky_label_0_image, sky_covar_0, sky_mean_0,
                        cv::COVAR_ROWS | cv::COVAR_NORMAL | cv::COVAR_SCALE);
    cv::Mat ic_s0;
    cv::invert(sky_covar_0, ic_s0, cv::DECOMP_SVD);

    cv::Mat ground_covar, ground_mean;
    cv::calcCovarMatrix(ground_image_non_zero, ground_covar, ground_mean,
                        cv::COVAR_ROWS | cv::COVAR_NORMAL | cv::COVAR_SCALE);
    cv::Mat ic_g;
    cv::invert(ground_covar, ic_g, cv::DECOMP_SVD);

    // Mahalanobis distance: Distancce between point and distribution
    cv::Mat sky_covar, sky_mean, ic_s;
    if (cv::Mahalanobis(sky_mean_0, ground_mean, ic_s0) > cv::Mahalanobis(sky_mean_1, ground_mean, ic_s1))
    {
        sky_mean = sky_mean_0;
        sky_covar = sky_covar_0;
        ic_s = ic_s0;
    }
    else
    {

        sky_mean = sky_mean_1;
        sky_covar = sky_covar_1;
        ic_s = ic_s1;
    }

    refined_border = std::vector<int>(border.size(), 0);
    for (size_t col = 0; col < border.size(); ++col)
    {
        double cnt = 0.0; // Number of sky pixel in i_th column
        for (int row = 0; row < border[col]; ++row)
        {
            // Calculate the Mahalanobis distance between each pixel in the original sky area and
            // each point in the corrected sky area
            cv::Mat ori_pix;
            img.row(row).col(static_cast<int>(col)).convertTo(ori_pix, sky_mean.type());
            double distance_s = cv::Mahalanobis(ori_pix, sky_mean, ic_s);
            double distance_g = cv::Mahalanobis(ori_pix, ground_mean, ic_g);

            if (distance_s < distance_g)
                ++cnt;
        }
        if (cnt < (border[col] / 2))
            refined_border[col] = 0;
        else
            refined_border[col] = border[col];
    }
}

bool SkyDetector::extract_sky(const cv::Mat &img, cv::Mat &sky_mask)
{
    int img_H = img.size[0];
    int img_W = img.size[1];

    std::cout << "Extract border...\n";
    std::vector<int> sky_border_optimal;
    extract_border_optimal(img, sky_border_optimal);

    if (!has_sky_region(sky_border_optimal, img_H / 30, img_H / 4, 2))
    {
        std::cout << "No sky area extracted" << std::endl;
        return false;
    }

    std::cout << "It has sky region...\n";
    if (has_partial_sky_region(sky_border_optimal, img_W / 3))
    {
        std::cout << "It has a partial sky region...\n";
        std::vector<int> border_new;
        std::cout << "Refining region...\n";
        refine_border(sky_border_optimal, img, border_new);
        sky_mask = make_sky_mask(img, border_new);
        return true;
    }

    sky_mask = make_sky_mask(img, sky_border_optimal);
    return true;
}

bool SkyDetector::extract_ground(const cv::Mat &img, cv::Mat &ground_mask)
{
    int img_H = img.size[0];
    int img_W = img.size[1];

    std::cout << "Extract border...\n";
    std::vector<int> sky_border_optimal;
    extract_border_optimal(img, sky_border_optimal);

    if (!has_sky_region(sky_border_optimal, img_H / 30, img_H / 4, 2))
    {
        std::cout << "No sky area extracted" << std::endl;
        return false;
    }

    std::cout << "It has sky region...\n";
    if (has_partial_sky_region(sky_border_optimal, img_W / 3))
    {
        std::cout << "It has a partial sky region...\n";
        std::vector<int> border_new;
        std::cout << "Refining region...\n";
        refine_border(sky_border_optimal, img, border_new);
        ground_mask = make_sky_mask(img, border_new, 0);
        return true;
    }

    ground_mask = make_sky_mask(img, sky_border_optimal, 0);
    return true;
}
