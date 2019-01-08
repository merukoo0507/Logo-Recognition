#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"

#include <iostream>
#include <windows.h>
#include <string>
#include <vector>
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include<iostream>
#include<fstream>
#include <time.h>

using namespace cv;
using namespace std;

vector <string> classname;
vector <Mat> vctRefImage;
vector <int> vctIdxRef;

//主要路徑
//放正規的logo圖檔
string refDir = "C:\\Qt\\workspace\\LogoRecognition\\RefImage";
//待測試的logo圖檔
string testDir = "C:\\Qt\\workspace\\LogoRecognition\\ROI";
//紀錄結果
char filename[] = "resultFlickrLogos.txt";

int getdir(string dir, vector<string> &files);
void dirName();
void DetectImageSetsKeypointDescriptor(vector <Mat> vctImg,
                                       vector < vector<KeyPoint> > &keypoints,
                                       vector < Mat > &descriptors,
                                       Ptr<FeatureDetector>& detector, Ptr<DescriptorExtractor>& descriptorExtractor);
static void crossCheckMatching( Ptr<DescriptorMatcher>& descriptorMatcher,
                         const Mat& descriptors1, const Mat& descriptors2,
                         vector<DMatch>& filteredMatches12, int knn=1 )
{
    filteredMatches12.clear();
    vector<vector<DMatch> > matches12, matches21;
    descriptorMatcher->knnMatch( descriptors1, descriptors2, matches12, knn );
    descriptorMatcher->knnMatch( descriptors2, descriptors1, matches21, knn );
    for( size_t m = 0; m < matches12.size(); m++ )
    {
        bool findCrossCheck = false;
        for( size_t fk = 0; fk < matches12[m].size(); fk++ )
        {
            DMatch forward = matches12[m][fk];

            for( size_t bk = 0; bk < matches21[forward.trainIdx].size(); bk++ )
            {
                DMatch backward = matches21[forward.trainIdx][bk];
                if( backward.trainIdx == forward.queryIdx )
                {
                    filteredMatches12.push_back(forward);
                    findCrossCheck = true;
                    break;
                }
            }
            if( findCrossCheck ) break;
        }
    }
}

double doIteration( const Mat& img1, Mat& img2,
                         vector<KeyPoint> &keypoints1, const Mat& descriptors1,
                         vector<KeyPoint> &keypoints2, const Mat& descriptors2,
                         Ptr<DescriptorMatcher>& descriptorMatcher,
                         double ransacReprojThreshold, BOWImgDescriptorExtractor &bowDE, bool show);

void DetectImageSetsKeypointDescriptor(vector <Mat> vctImg,
                                       vector < vector<KeyPoint> > &keypoints,
                                       vector < Mat > &descriptors,
                                       Ptr<FeatureDetector>& detector, Ptr<DescriptorExtractor>& descriptorExtractor);

void BowTesting();

int main()
{
    //建立vector, 正規logo圖片的集合
    dirName();
    //BOW, 開始測試
    BowTesting();
    return 0;
}

void BowTesting()
{
    //====BOOW===========================================================================================
    Ptr<FeatureDetector> detector(new SurfFeatureDetector());;
    Ptr<DescriptorExtractor> descriptorExtractor(new SurfDescriptorExtractor);;
    Ptr<DescriptorMatcher> descriptorMatcher = DescriptorMatcher::create("FlannBased");//"BruteForce");
    vector < vector<KeyPoint> > refKeypoints(vctRefImage.size());
    vector < Mat > refDescriptors(vctRefImage.size());
    DetectImageSetsKeypointDescriptor(vctRefImage, refKeypoints, refDescriptors, detector, descriptorExtractor);

    Mat featuresUnclustered;
    for (int i = 0; i < (int)refDescriptors.size(); i++)
    {
        featuresUnclustered.push_back(refDescriptors[i]);
    }
    int dictionarySize = 2500;
    TermCriteria tc(CV_TERMCRIT_ITER, 100, 0.001);
    int retries = 1;
    int flags = KMEANS_PP_CENTERS;
    BOWKMeansTrainer bowTrainer(dictionarySize, tc, retries, flags);
    Mat mDictionary = bowTrainer.cluster(featuresUnclustered);

    BOWImgDescriptorExtractor bowDE(descriptorExtractor, descriptorMatcher);//create BoW descriptor extractor
    bowDE.setVocabulary(mDictionary);

    cout << "TESTING" << endl;
    //====TESTING===========================================================================================
    int ransacReprojThreshold = 3;
    int correct = 0, error = 0;
    fstream fp;
    fp.open(filename, ios::out);
    if (!fp)
       cout << "Fail." << endl;

    clock_t startTime, endTime;
    double runningTime=0;
    startTime=clock();

    const int cINT = (int)classname.size();
    int table[cINT][cINT];
    for (int i=0; i<cINT; i++)
        for (int j=0; j<cINT; j++)
            table[i][j] = 0;
    for (int c = 0; c < cINT; c++)  //classname.size()
    {
        vector<string> files;
        vector <Mat> vctTestImage;

        string s = testDir + "\\" + classname[c];
        getdir(s, files);
        for (int i = 0; i < (int)files.size(); i++)
        {
            s = testDir + "\\" + classname[c]+"\\"+files[i];
            Mat img = imread(s, CV_LOAD_IMAGE_GRAYSCALE);
            if (img.empty()) continue;
            else
            {
                if (img.rows < img.cols)
                {
                    float rr = 150.0 / img.rows;
                    int ww = img.cols * rr;
                    resize(img, img, Size(ww, 150));
                }
                else{
                    float rr = 150.0 / img.cols;
                    int hh = img.rows * rr;
                    resize(img, img, Size(150, hh));
                }
            }
            vctTestImage.push_back(img);
        }
        files.clear();

        vector < vector<KeyPoint> > testKeypoints(vctTestImage.size());
        vector < Mat > testDescriptors(vctTestImage.size());
        DetectImageSetsKeypointDescriptor(vctTestImage, testKeypoints, testDescriptors, detector, descriptorExtractor);

        int clsCorrect = 0;
        int clsError = 0;
        for (int i = 0; i < (int)vctTestImage.size(); i++)
        {
            if (testKeypoints[i].size()==0) continue;

            float minDistance = 9e+9;
            int minIndex = 0;
            for (int j = 0; j < (int)vctRefImage.size(); j++)
            {
                double disValue = doIteration(vctTestImage[i], vctRefImage[j],
                                              testKeypoints[i], testDescriptors[i],
                                              refKeypoints[j], refDescriptors[j],
                                              descriptorMatcher, ransacReprojThreshold, bowDE, false);
                if (disValue < minDistance)
                {
                    minDistance = disValue;
                    minIndex = j;
                }
            }
//            Classify result
            doIteration(vctTestImage[i], vctRefImage[minIndex],
                        testKeypoints[i], testDescriptors[i],
                        refKeypoints[minIndex], refDescriptors[minIndex],
                        descriptorMatcher, ransacReprojThreshold, bowDE, false);    //true

//			printf("%d\n", i);
            table[c][vctIdxRef[minIndex]]++;
            if (vctIdxRef[minIndex] == c)
            {
                correct++;
                clsCorrect++;
            }
            else{
                error++;
                clsError++;
            }
            //the result of testImage(i)
            cout << "t(" << i << "):" << vctIdxRef[minIndex] << endl;
        }
        cout << endl;
//        fp << c << ": co:" << clsCorrect  << " , err:" << clsError << endl;
        cout << c << ": co:" << clsCorrect  << " , err:" << clsError << endl;
        cout << c << ": ";
        fp << c << ": ";
        for (int j=0; j<cINT; j++)
        {
            cout << table[c][j] << " ";
            fp << table[c][j] << " ";
        }
        cout << endl;
        fp << endl;
    }

    endTime=clock();
    runningTime = (double)(endTime-startTime)/CLOCKS_PER_SEC;
    cout << runningTime << endl;
    cout << "atime:" << (runningTime / (double)(error+correct)) << endl;

    cout << "total_co:" << correct  << " , total_err:" << error << endl;
    double p = (double)correct/(double)(error+correct);
    cout << "prediction:" << p << endl;

    fp << "atime:" << (runningTime / (double)(error+correct)) << endl;
    fp << "runningTime:" << runningTime << endl;

    fp << "total_co:" << correct  << " , total_err:" << error << endl;
    fp << "prediction:" << p << endl;

    fp.close();
}

double doIteration( const Mat& img1, Mat& img2,
                         vector<KeyPoint> &keypoints1, const Mat& descriptors1,
                         vector<KeyPoint> &keypoints2, const Mat& descriptors2,
                         Ptr<DescriptorMatcher>& descriptorMatcher,
                         double ransacReprojThreshold, BOWImgDescriptorExtractor &bowDE, bool show)
{
    //====Cross Check and filter==============================================================
    vector<DMatch> filteredMatches;
    crossCheckMatching( descriptorMatcher, descriptors1, descriptors2, filteredMatches, 1 );
    //-----------------------------------------------------------------------------------------
    if (filteredMatches.size() > 3)
    {
        vector<int> queryIdxs(filteredMatches.size());
        vector<int> trainIdxs(filteredMatches.size());
        vector <KeyPoint> filterkeypoints1(filteredMatches.size());
        vector <KeyPoint> filterkeypoints2(filteredMatches.size());
        for (size_t i = 0; i < filteredMatches.size(); i++)
        {
            queryIdxs[i] = filteredMatches[i].queryIdx;//query = descriptors1
            trainIdxs[i] = filteredMatches[i].trainIdx;//train = descriptors2
            filterkeypoints1[i] = keypoints1[filteredMatches[i].queryIdx];
            filterkeypoints2[i] = keypoints2[filteredMatches[i].trainIdx];
        }

        //====Find homography (RANSAC) matrix======================================================
        vector<Point2f> points1;
        vector<Point2f> points2;
        KeyPoint::convert(keypoints1, points1, queryIdxs);//KeyPoint convert to Point2f
        KeyPoint::convert(keypoints2, points2, trainIdxs);//KeyPoint convert to Point2f

        Mat H12;
        if (ransacReprojThreshold >= 0)
        {
            H12 = findHomography(Mat(points1), Mat(points2), CV_RANSAC, ransacReprojThreshold);
        }
        //-----------------------------------------------------------------------------------------
        //====決定那些配對的點座標位置是相近的=========================================================
        Mat points1t;
        vector<char> matchesMask(filteredMatches.size(), 0);
        perspectiveTransform(Mat(points1), points1t, H12);

        double maxInlierDist = ransacReprojThreshold < 0 ? 3 : ransacReprojThreshold;

        for (int i1 = (points1.size() - 1); i1 >= 0; i1--)
        {
            double l2dist = norm(points2[i1] - points1t.at<Point2f>(i1, 0));
            if (l2dist <= maxInlierDist) // inlier
            {
                matchesMask[i1] = 1;

            }
            else
            {
                filterkeypoints1.erase(filterkeypoints1.begin() + i1);
                filterkeypoints2.erase(filterkeypoints2.begin() + i1);
            }

        }

        if (filterkeypoints1.size() > 0)
        {
            if (show)
            {
                // draw inliers
                Mat drawImg;
                drawMatches(img1, keypoints1, img2, keypoints2, filteredMatches, drawImg, CV_RGB(0, 255, 0), CV_RGB(0, 0, 255), matchesMask);

                //-----------------------------------------------------------------------------------------

                //====圖像的校正============================================================================
                Mat warpImg = img2.clone();
                warpPerspective(img1, warpImg, H12, warpImg.size());
                //			warpImg = warpImg.rowRange(0, img1.rows).colRange(0, img1.cols).clone();
                //-----------------------------------------------------------------------------------------
                imshow("warpPerspective", warpImg);
                imshow("correspondences", drawImg);
            }

            //====Computing BOW feature and distance===================================================
            Mat bowDescriptor1;
            Mat bowDescriptor2;
            bowDE.compute(img1, filterkeypoints1, bowDescriptor1);//extract SURF
            bowDE.compute(img2, filterkeypoints2, bowDescriptor2);//extract SURF
            return (norm(bowDescriptor1, bowDescriptor2, NORM_L2));
            //-----------------------------------------------------------------------------------------
        }
        else
            return 9e+9;
    }
    else
        return 9e+9;
}

void DetectImageSetsKeypointDescriptor(vector <Mat> vctImg,
                                       vector < vector<KeyPoint> > &keypoints,
                                       vector < Mat > &descriptors,
                                       Ptr<FeatureDetector>& detector, Ptr<DescriptorExtractor>& descriptorExtractor)
{
    for (int i = 0; i < (int)vctImg.size(); i++)
    {
        detector->detect(vctImg[i], keypoints[i]);
        descriptorExtractor->compute(vctImg[i], keypoints[i], descriptors[i]);
    }
}

int getdir(string dir, vector<string> &files){
    DIR *dp;//創立資料夾指標
    struct dirent *dirp;
    if((dp = opendir(dir.c_str())) == NULL){
        cout << "Error(" << errno << ") opening " << dir << endl;
        return errno;
    }
    while((dirp = readdir(dp)) != NULL){//如果dirent指標非空
        files.push_back(string(dirp->d_name));//將資料夾和檔案名放入vector
    }
    closedir(dp);//關閉資料夾指標
    return 0;
}

void dirName(){
    vector<string> files;

    getdir(refDir, files);
    for (int i=2; i<(int)files.size(); i++)
    {
        string s = refDir + "\\" + files[i];
        cout << files[i] << endl;
        classname.push_back(files[i]);
    }
    cout << "--------classname/End--------" << endl;
    files.clear();

    for (int c = 0; c < (int)classname.size(); c++)
    {
        string s = refDir + "\\" + classname[c];
        getdir(s, files);
        for (int i = 0; i < (int)files.size(); i++)
        {
            s = refDir + "\\" + classname[c]+"\\"+files[i];
            Mat img = imread(s, CV_LOAD_IMAGE_GRAYSCALE);
            if (img.empty()) continue;
            vctRefImage.push_back(img);
            vctIdxRef.push_back(c);
            cout << c << ":" << s << endl;
        }
        files.clear();
    }
    cout << "--------vctRefImage&vctIdxRef/End--------" << endl;
    system("cls");
}
