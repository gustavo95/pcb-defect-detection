#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

cv::Mat imgAlignment(Mat &img1, Mat &img2)
{
    const int warp_mode = MOTION_EUCLIDEAN;

    // Set a 2x3 or 3x3 warp matrix depending on the motion model.
    Mat warp_matrix;

    // Initialize the matrix to identity
    if ( warp_mode == MOTION_HOMOGRAPHY )
        warp_matrix = Mat::eye(3, 3, CV_32F);
    else
        warp_matrix = Mat::eye(2, 3, CV_32F);

    // Specify the number of iterations.
    int number_of_iterations = 5000;

    // Specify the threshold of the increment
    // in the correlation coefficient between two iterations
    double termination_eps = 1e-10;

    // Define termination criteria
    TermCriteria criteria (TermCriteria::COUNT+TermCriteria::EPS, number_of_iterations, termination_eps);

    // Run the ECC algorithm. The results are stored in warp_matrix.
    findTransformECC(
                     img1,
                     img2,
                     warp_matrix,
                     warp_mode,
                     criteria
                 );

    // Storage for warped image.
    Mat img2_aligned;

    if (warp_mode != MOTION_HOMOGRAPHY)
        // Use warpAffine for Translation, Euclidean and Affine
        warpAffine(img2, img2_aligned, warp_matrix, img1.size(), INTER_LINEAR + WARP_INVERSE_MAP);
    else
        // Use warpPerspective for Homography
        warpPerspective (img2, img2_aligned, warp_matrix, img1.size(),INTER_LINEAR + WARP_INVERSE_MAP);

    return img2_aligned;
}

cv::Mat imgPreprocessor(Mat &img)
{
    Mat result;
    threshold(img, result, 127, 255, CV_THRESH_BINARY);
    //result = Scalar(255) - result;
    return result;
}

vector<Mat> imgSegmentation(Mat &img)
{
    vector<Mat> segments;

    //Hole detection
    Mat kernel1 = getStructuringElement(0, Size(17, 17), Point(8, 8));
    Mat detHole1, detHole;
    morphologyEx(img, detHole1, MORPH_CLOSE, kernel1);
    bitwise_xor(detHole1, img, detHole);

    //Square pad segmentation
    Mat kernel2 = getStructuringElement(0, Size(71, 71), Point(35, 35));
    Mat squarePad1, squarePad;
    morphologyEx(detHole1, squarePad1, MORPH_OPEN, kernel2);
    dilate(squarePad1, squarePad, img);
    imshow("img", squarePad1);
    imshow("img2", detHole1);

    return segments;
}

int main()
{
    namedWindow("test", WINDOW_KEEPRATIO);
    namedWindow("reference", WINDOW_KEEPRATIO);
    namedWindow("img", WINDOW_KEEPRATIO);
    namedWindow("img2", WINDOW_KEEPRATIO);

    Mat testImg = imread("test.png", CV_LOAD_IMAGE_GRAYSCALE);
    Mat refImg = imread("ref.png", CV_LOAD_IMAGE_GRAYSCALE);
    Mat img, img2;

    if(!testImg.data)
    {
        cout <<  "Could not open or find the test image" << std::endl ;
        return -1;
    }
    if(!refImg.data)
    {
        cout <<  "Could not open or find the reference image" << std::endl ;
        return -1;
    }


    resize(refImg, refImg, testImg.size(), 0, 0, INTER_LINEAR);

    refImg = imgAlignment(testImg, refImg);

    testImg = imgPreprocessor(testImg);
    refImg = imgPreprocessor(refImg);

    //img = testImg - refImg;
    //img2 = refImg - testImg;

    //Mat kernel = getStructuringElement(0, Size(7, 7), Point(3, 3));
    //morphologyEx(img, img, MORPH_OPEN, kernel);
    //morphologyEx(img2, img2, MORPH_OPEN, kernel);

    imgSegmentation(testImg);

    //-------Testing--------
    /*
    int morph_sizex = 8;
    int morph_sizey = 8;
    int const max_kernel_size = 50;
    createTrackbar( "Kernel size x:", "img", &morph_sizex, max_kernel_size, 0);
    createTrackbar( "Kernel size y:", "img", &morph_sizey, max_kernel_size, 0);

    //Hole detection
    Mat kernel1 = getStructuringElement(0, Size(17, 17), Point(8, 8));
    Mat detHole1, detHole;
    morphologyEx(testImg, detHole1, MORPH_CLOSE, kernel1);
    bitwise_xor(detHole1, testImg, detHole);

    //Square pad segmentation
    Mat kernel2 = getStructuringElement(0, Size(71, 71), Point(35, 35));
    Mat squarePad1, squarePad;
    morphologyEx(detHole1, squarePad1, MORPH_OPEN, kernel2);
    //dilate(squarePad1, squarePad, kernel2);
    //imshow("img", squarePad1);
    imshow("img2", squarePad1);
    dilate(squarePad1, img, testImg);
    */

    for(;;)
    {
        //Mat kernel = getStructuringElement(0, Size(2*morph_sizex+1, 2*morph_sizey+1), Point( morph_sizex, morph_sizey));
        //morphologyEx(detHole1, img, MORPH_OPEN, kernel);

        imshow("test", testImg);
        imshow("reference", refImg);
        //imshow("img", img);
        //imshow("img2", img2);
        if((char) waitKey(1) == 'q') break;
    }

    return 0;
}
