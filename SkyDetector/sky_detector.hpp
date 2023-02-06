#include <opencv2/opencv.hpp>
#include <vector>

class SkyDetector
{
public:
    SkyDetector() = default;
    ~SkyDetector() = default;

    bool extract_sky(const cv::Mat &img, cv::Mat &sky_mask);
    bool extract_ground(const cv::Mat &img, cv::Mat &sky_mask);

private:
    double f_thres_sky_max = 600;
    double f_thres_sky_min = 5;
    double f_thres_sky_search_step = 5;

    // Extract image sky area
    void extract_border_optimal(const cv::Mat &img, std::vector<int> &sky_border_optimal, cv::Mat& gradient_info_map);
    void calculate_border_naive(const cv::Mat& gradient_info_map, std::vector<int> &sky_border_naive);
    void extract_image_gradient(const cv::Mat &img, cv::Mat& gradient_img);
    void extract_border(const cv::Mat& gradient, double t, std::vector<int>& b);
    double calculate_sky_energy(const std::vector<int> &border, const cv::Mat &img);
    double calculate_sky_energy_gray(const std::vector<int> &border, const cv::Mat &img);

    cv::Mat make_sky_mask(const cv::Mat &img, const std::vector<int> &border, int type=1);
    bool has_sky_region(const std::vector<int>& border, double &border_mean, double &border_diff_mean,
                        double thresh_1, double thresh_2, double thresh_3);
    bool has_partial_sky_region(const std::vector<int> &border, double thresh_1);
    void refine_border(const std::vector<int> &border,  const cv::Mat &img, std::vector<int> &refined_border);
    bool test_ground_mask(const cv::Mat& mask, const std::vector<int>& border);
    void cleanup_border(std::vector<int>& border);
};
