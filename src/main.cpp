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
    threshold(img, result, 47, 255, CV_THRESH_BINARY);
    return result;
}

vector<Mat> imgSegmentation(Mat &img)
{
    vector<Mat> segments;
    segments.reserve(4);

    //Hole detection
    Mat kernel1 = getStructuringElement(2, Size(4, 4), Point(2, 2));
    Mat detHole1, detHole;
    morphologyEx(img, detHole1, MORPH_CLOSE, kernel1);
    //detHole1 = img;
    bitwise_xor(detHole1, img, detHole);

    //Square pad segmentation
    Mat kernel2 = getStructuringElement(0, Size(31, 31), Point(15, 15));
    Mat squarePad1;
    morphologyEx(detHole1, squarePad1, MORPH_OPEN, kernel2);

    //Hole pad segmentation
    Mat holePad1, holePad2;
    Mat kernel3 = getStructuringElement(0, Size(7, 7), Point(3, 3));
    bitwise_xor(detHole1, squarePad1, holePad1);
    morphologyEx(holePad1, holePad2, MORPH_OPEN, kernel3);

    //Rectangular pad segmentation
    Mat rectPad1, rectPad2;
    Mat kernel4 = getStructuringElement(2, Size(5, 5), Point(2, 2));
    bitwise_xor(holePad1, holePad2, rectPad1);
    morphologyEx(rectPad1, rectPad2, MORPH_OPEN, kernel4);

    segments.push_back(squarePad1); //Square
    segments.push_back(holePad2);   //Hole
    segments.push_back(rectPad2);   //Line

    return segments;
}

int main()
{
    namedWindow("img1", WINDOW_KEEPRATIO);
    namedWindow("img2", WINDOW_KEEPRATIO);
    namedWindow("img3", WINDOW_KEEPRATIO);
    namedWindow("img4", WINDOW_KEEPRATIO);
    namedWindow("img5", WINDOW_KEEPRATIO);

    Mat testImg = imread("test_pcb.png", CV_LOAD_IMAGE_GRAYSCALE);
    Mat refImg = imread("ref_pcb.png", CV_LOAD_IMAGE_GRAYSCALE);

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

    //refImg = imgAlignment(testImg, refImg);

    testImg = imgPreprocessor(testImg);
    refImg = imgPreprocessor(refImg);

    vector<Mat> refSegments = imgSegmentation(refImg);
    vector<Mat> testSegments = imgSegmentation(testImg);

    //Positive - Missing
    //Negative - Execive
    Mat G13, G21, G22, G42, G43;
    G13 = testSegments[0] - refSegments[0];
    G21 = testSegments[1] - refSegments[1];
    G22 = refSegments[1] - testSegments[1];
    G42 = testSegments[2] - refSegments[2];
    G43 = refSegments[2] - testSegments[2];

    cvtColor(testImg, testImg, CV_GRAY2BGR);
    vector<Vec4i> lines;
    HoughLinesP(G13, lines, 1, CV_PI/180, 0, 0, 0);
    for(size_t i = 0; i < lines.size(); i++)
    {
        Vec4i l = lines[i];
        line(testImg, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 2, 1);
    }

    HoughLinesP(G21, lines, 1, CV_PI/180, 8, 8, 0);
    for(size_t i = 0; i < lines.size(); i++)
    {
        Vec4i l = lines[i];
        line(testImg, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 2, 1);
    }

    HoughLinesP(G22, lines, 1, CV_PI/180, 0, 0, 0);
    for(size_t i = 0; i < lines.size(); i++)
    {
        Vec4i l = lines[i];
        line(testImg, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 2, 1);
    }

    HoughLinesP(G42-G22, lines, 1, CV_PI/180, 0, 0, 0);
    for(size_t i = 0; i < lines.size(); i++)
    {
        Vec4i l = lines[i];
        line(testImg, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 2, 1);
    }

    HoughLinesP(G43, lines, 1, CV_PI/180, 0, 0, 0);
    for(size_t i = 0; i < lines.size(); i++)
    {
        Vec4i l = lines[i];
        line(testImg, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 2, 1);
    }

    Mat img = testImg.clone();

    int x = 8;
    int y = 8;
    int z = 8;
    createTrackbar( "x:", "img5", &x, 200, 0);
    createTrackbar( "y:", "img5", &y, 200, 0);
    createTrackbar( "z:", "img5", &z, 200, 0);

    for(;;)
    {
        imshow("img1", G13);
        imshow("img2", G21);
        imshow("img3", G22);
        imshow("img4", G42-G22);
        imshow("img5", G43);
        imshow("img7", testSegments[2]);
        imshow("img8", refSegments[2]);

        img = testImg.clone();

        imshow("img6", img);

        if((char) waitKey(1) == 'q') break;
    }

    return 0;
}
