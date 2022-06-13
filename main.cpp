#include <iostream>
#include <opencv2/opencv.hpp>
#include <numeric>
#include <chrono>
#include <ctime>
using namespace cv;
using namespace cv::ml;
using namespace std;
// parameter
const short int minarea = 50;               //灯条的最小识别面积
const short int wid_hei = 3;                //灯条的最小高宽比
const short int maxarea = 1200;             //灯条的最大识别面积
const short int max_dist_x_ratio = 3;       //两个灯条X坐标与灯条高度的最大比值
const short int min_dist_x_ratio = 1;       //两个灯条X坐标与灯条高度的最小比值
const float parallel_tan = 0.3;             //两个灯条夹角的最大正切值
const short int mean_area_divisor = 3;      //两个灯条面积的差 < 两个灯条面积的和/mean_area_divisor
const short int bianry_threshold = 100;     //图像二值化的阈值
#define YELLOW                              //灯条的颜色  para: RED  YELLOW   BLUE

double getDistance(Point A, Point B)
{
    double dis;
    dis = pow((A.x - B.x), 2) + pow((A.y - B.y), 2);
    return sqrt(dis);
}

double getDistance(double a, double b, double c)
{
    double dis;
    dis = pow(a, 2) + pow(b, 2) + pow(c, 2);
    return sqrt(dis);
}

Point2i getMidpoint(Point A, Point B)
{
    Point2i P;
    P.x = (A.x + B.x) / 2;
    P.y = (A.y + B.y) / 2;
    return P;
}

Point2i getMidpoint(Point A, Point B, Point C, Point D)
{
    Point2i P;
    P.x = (A.x + B.x + C.x + D.x) / 4;
    P.y = (A.y + B.y + C.y + D.y) / 4;
    return P;
}

typedef struct
{
    string type;
    string data;
    vector<Point> location;
} decodedObject;

double display(Mat &im, vector<decodedObject> &decodedObjects, Mat cam, Mat dis)
{

    for (int i = 0; i < decodedObjects.size(); i++)
    {
        vector<Point> points = decodedObjects[i].location;
        vector<Point> hull;
        if (points.size() > 4)
            convexHull(points, hull);
        else
            hull = points;
        vector<Point2f> pnts;
        int n = hull.size();
        for (int j = 0; j < n; j++)
        {
            line(im, hull[j], hull[(j + 1) % n], Scalar(255, 0, 0), 3);
            pnts.push_back(Point2f(hull[j].x, hull[j].y));
        }
#define HALF_LENGTH 33.75
#define HALF_HIGH 13.25
        vector<Point3f> obj = vector<Point3f>{
            cv::Point3f(-HALF_LENGTH, -HALF_HIGH, 0), // tl
            cv::Point3f(HALF_LENGTH, -HALF_HIGH, 0),  // tr
            cv::Point3f(HALF_LENGTH, HALF_HIGH, 0),   // br
            cv::Point3f(-HALF_LENGTH, HALF_HIGH, 0)   // bl
        };
        cv::Mat rVec = cv::Mat::zeros(3, 1, CV_64FC1); // init rvec
        cv::Mat tVec = cv::Mat::zeros(3, 1, CV_64FC1); // init tvec
        if (pnts.size() == 4)
        {
            solvePnP(obj, pnts, cam, dis, rVec, tVec);
            double dist;
            dist = getDistance(tVec.at<double>(0, 0), tVec.at<double>(1, 0), tVec.at<double>(2, 0));
            return dist;
        }
    }
}

int main()
{
    cv::VideoCapture capture("/home/rm/code/装甲板_1.avi");
    cv::Mat frame;
    Mat distCoeffs;
    Mat cameraMatrix;
    FileStorage fs("out_camera_data.xml", FileStorage::READ);
    fs["camera_matrix"] >> cameraMatrix;
    fs["distortion_coefficients"] >> distCoeffs;
    Point2i past_point[3], predict_point;
    for (int i = 0; i < 3; i++)
    {
        past_point[i].x = -1;
        past_point[i].y = -1;
    }
    while (capture.read(frame))
    {
        double t = getTickCount();
        vector<Mat> imgChannels;
        split(frame, imgChannels);
//获得目标颜色图像的二值图
#ifdef RED
        Mat midImage2 = imgChannels.at(2) - imgChannels.at(1);
#endif
#ifdef YELLOW
        Mat midImage2 = imgChannels.at(2) - imgChannels.at(0);
#endif
#ifdef BLUE
        Mat midImage2 = imgChannels.at(0) - imgChannels.at(2);
#endif
        //二值化，背景为黑色，图案为白色
        threshold(midImage2, midImage2, bianry_threshold, 255, cv::THRESH_BINARY);
        int structElementSize = 2;
        Mat element = getStructuringElement(MORPH_RECT, Size(2 * structElementSize + 1, 2 * structElementSize + 1), Point(structElementSize, structElementSize));
        dilate(midImage2, midImage2, element);
        structElementSize = 3;
        element = getStructuringElement(MORPH_RECT, Size(2 * structElementSize + 1, 2 * structElementSize + 1), Point(structElementSize, structElementSize));
        morphologyEx(midImage2, midImage2, MORPH_CLOSE, element);
        vector<vector<Point>> lightContours;
 
        //找轮廓
        findContours(midImage2.clone(), lightContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        vector<vector<Point>> contours2;
        vector<Vec4i> hierarchy2;
        findContours(midImage2, contours2, hierarchy2, RETR_CCOMP, CHAIN_APPROX_SIMPLE);

        RotatedRect rect_tmp2;
        short int unit = 0;
        Point2f target;
        if (hierarchy2.size())
        {
            vector<vector<Point2i>> prePionts;
            vector<int> CountArea;
            for (int i = 0; i >= 0; i = hierarchy2[i][0])
            {
                rect_tmp2 = minAreaRect(contours2[i]);
                Point2f P[4];
                //将矩形的四个点保存在P中
                rect_tmp2.points(P);
                //为透视变换做准备
                Point2i srcRect[4];
                double width;
                double height;
                //矫正提取的叶片的宽高
                height = getDistance(P[0], P[1]);
                width = getDistance(P[1], P[2]);
                if (width < height)
                {
                    srcRect[0] = P[0];
                    srcRect[1] = P[1];
                    srcRect[2] = P[2];
                    srcRect[3] = P[3];
                }
                else
                {
                    swap(width, height);
                    srcRect[0] = P[1];
                    srcRect[1] = P[2];
                    srcRect[2] = P[3];
                    srcRect[3] = P[0];
                }
                double area = height * width;
                if (area > minarea && height / width > wid_hei)
                {
                    unit += height;
                    vector<Point2i> B;
                    B.push_back(getMidpoint(srcRect[0], srcRect[3]));
                    B.push_back(getMidpoint(srcRect[1], srcRect[2]));
                    B.push_back(srcRect[1]);
                    B.push_back(srcRect[2]);
                    prePionts.push_back(B);
                    B.clear();
                    CountArea.push_back(area);
                }
            }
            if (CountArea.size() > 1)
            {
                unit = unit / CountArea.size();
                if (prePionts.size() >= 2)
                {
                    bool flag = 1;
                    for (int i = 0; i < prePionts.size() - 1 && flag; i++)
                    {
                        for (int j = i + 1; j < prePionts.size() && flag; j++)
                        {
                            if (abs(prePionts[i][0].x - prePionts[j][0].x) > unit * min_dist_x_ratio && abs(prePionts[i][0].x - prePionts[j][0].x < max_dist_x_ratio * unit) && abs(CountArea[i] - CountArea[j]) < (CountArea[i] + CountArea[j]) / mean_area_divisor)
                            {
                                short x1 = abs(prePionts[i][0].x - prePionts[i][1].x);
                                short x2 = abs(prePionts[j][0].x - prePionts[j][1].x);
                                short y1 = abs(prePionts[i][0].y - prePionts[i][1].y);
                                short y2 = abs(prePionts[j][0].y - prePionts[j][1].y);
                                if ((x1 * x2 + y1 * y2) != 0 && abs((x1 * y2 - x2 * y1) / (x1 * x2 + y1 * y2)) < parallel_tan)
                                {
                                    target = getMidpoint(prePionts[i][0], prePionts[i][1], prePionts[j][0], prePionts[j][1]);
                                    circle(frame, prePionts[i][0], 3, Scalar(0, 255, 0), -1);
                                    circle(frame, prePionts[i][1], 3, Scalar(0, 255, 0), -1);
                                    circle(frame, prePionts[j][0], 3, Scalar(0, 255, 0), -1);
                                    circle(frame, prePionts[j][1], 3, Scalar(0, 255, 0), -1);
                                    circle(frame, target, 3, Scalar(255, 0, 0), -1);
                                    vector<Point2f> vertex;
                                    vertex.push_back(prePionts[i][0]);
                                    vertex.push_back(prePionts[i][1]);
                                    vertex.push_back(prePionts[j][0]);
                                    vertex.push_back(prePionts[j][1]);
                                    vertex.push_back(target);
                                    decodedObject obj;
                                    for (int i = 0; i < 4; i++) //左上角第一个，顺时针排序
                                    {
                                        if (vertex[i].x < vertex[4].x && vertex[i].y < vertex[4].y)
                                            obj.location.push_back(vertex[i]);
                                    }
                                    for (int i = 0; i < 4; i++)
                                    {
                                        if (vertex[i].x > vertex[4].x && vertex[i].y < vertex[4].y)
                                            obj.location.push_back(vertex[i]);
                                    }
                                    for (int i = 0; i < 4; i++)
                                    {
                                        if (vertex[i].x > vertex[4].x && vertex[i].y > vertex[4].y)
                                            obj.location.push_back(vertex[i]);
                                    }
                                    for (int i = 0; i < 4; i++)
                                    {
                                        if (vertex[i].x < vertex[4].x && vertex[i].y > vertex[4].y)
                                            obj.location.push_back(vertex[i]);
                                    }

                                    vector<decodedObject> decodedObjects;
                                    decodedObjects.push_back(obj);

                                    double dist = display(frame, decodedObjects, cameraMatrix, distCoeffs);
                                    flag = 0;
                                    if (past_point[2].x != -1 && past_point[2].y != -1)
                                    {
                                        predict_point.x = target.x + ((target.x - past_point[0].x) * 1 + (past_point[0].x - past_point[1].x) * 4 + (past_point[1].x - past_point[2].x) * 1) / 6;
                                        predict_point.y = target.y + ((target.y - past_point[0].y) * 1 + (past_point[0].y - past_point[1].y) * 4 + (past_point[1].y - past_point[2].y) * 1) / 6;
                                        circle(frame, predict_point, 3, Scalar(0, 0, 255), -1);
                                    }
                                    past_point[2].x = past_point[1].x;
                                    past_point[2].y = past_point[1].y;
                                    past_point[1].x = past_point[0].x;
                                    past_point[1].y = past_point[0].y;
                                    past_point[0].x = target.x;
                                    past_point[0].y = target.y;
                                }
                            }
                        }
                    }
                }
            }
        }
        double time = (double)(getTickCount() - t) / getTickFrequency();
        cout << "FPS:" << 1 / time << endl;
    }
}
