/*
 * Copyright (c) 2017 , 许泽平(Xu Zeping) & 祁俊昆(Qi Junkun), All rights reserved.
 * You can use this source code freely except business
 * 
 * This is a part of source code of our course project. In this project,we try to
 * use some classical algorithm of face recognition and evaluate their ability.
 *
 * If you want to use our code,please ensure you've already installed opencv and 
 * compile the cpp files with pkg-config
 *
 * This file is an implementation of Fisherfaces.
 *
 * Fisherfaces is based on an idea 
 * that same classes should cluster tightly together, while different classes are 
 * as far away as possible from each other in the lower-dimensional representation.
 *
 * Written by Qi Junkun
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <string>
#include <vector>
 
#include <opencv2/core/core.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

/*
 * read_csv - Read images from files and divide them into training set and test
 * set according to the number of iamges of per person in the training set
 */
static int read_csv(const string& filename, vector<Mat>& train_images,
              vector<int>& trian_labels, vector<Mat>& test_images,
              vector<int>& test_labels, int trian_number, char separator = ';');
/*
 * norm_0_255 - Create and return normalized image
 */
static Mat norm_0_255(InputArray _src);

int main(int argc, const char *argv[])
{
	/*
     * check for legal command line argument
     */
    if(argc != 3){
        cout << "usage: ./" << argv[0] << " <csv.ext>  <number>" << endl;
        exit(1);
    }
    /*
     * get the path of CSV and the number of images of per person in
     * the training set
     */
    string fn_csv = string(argv[1]);
    int trian_number = atoi(argv[2]);
    vector<Mat> trian_images, test_images;
    vector<int> trian_labels, test_labels;
    int test_number = 0;
    /*
     * read in the data which can fail if no valid
     */
    try{
        test_number = read_csv(fn_csv, trian_images, trian_labels,
        	                   test_images, test_labels, trian_number);
    }catch (cv::Exception& e) {
        cerr << "Error opening file \"" << fn_csv << "\". Reason: "
             << e.msg << endl;
        exit(1);
    }
    /*
     * note the number of images predicted correctly
     */
    int correctcnt = 0;
    for(int i = 0; i < 40*test_number; i++){
    	/*
         * select testsample from the test set
         */
    	Mat testSample = test_images[i];
        int testLabel = test_labels[i];
        /*
         * create an Eigenfaces model with the traing set
         * full PCA here 
         */
        Ptr<FaceRecognizer> model = createFisherFaceRecognizer();
        model->train(trian_images, trian_labels);
        /*
         * predicts the label of the test samplle and compare it with
         * its true label
         * note the confidence in the meanwhile
         */
        int predictedLabel = model->predict(testSample);
        /*
         * note and output the result
         */
        string rst = format("Predicted / Actual = %d / %d.", 
        	predictedLabel, testLabel);
        cout << rst << endl;
        if(predictedLabel == testLabel)
        	correctcnt++;
    }
    double correctrate = (double)correctcnt / (40*test_number);
    string rst = format("Correct rate = %d / %d = %lf",
    	                correctcnt, 40*test_number, correctrate);
    cout << rst << endl;
	return 0;
}

static int read_csv(const string& filename, vector<Mat>& trian_images,
              vector<int>& trian_labels, vector<Mat>& test_images,
              vector<int>& test_labels, int trian_number, char separator)
{
	std::ifstream file(filename.c_str(), ifstream::in);
    /*
     * check for legal filename
     */
    if (!file) {
        string error_message = "No valid input file.";
        CV_Error(CV_StsBadArg, error_message);
    }
    /*
     * divide images into training set and test set according to the
     * number of images of per person in the training set
     */
    string line, path, classlabel;
    int test_number = 10-trian_number;
    /*
     * 40 person in total
     */
    int trian[40];
    memset(trian, 0 ,sizeof(trian));
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) {
        	int tmplabel = atoi(classlabel.c_str());
        	/*
        	 * add to triang set
        	 */
        	if(trian[tmplabel] < trian_number){
        		trian_images.push_back(imread(path, 0));
                trian_labels.push_back(atoi(classlabel.c_str()));
                trian[tmplabel]++;
        	}
        	/*
        	 * add to test set
        	 */
            else{
            	test_images.push_back(imread(path, 0));
                test_labels.push_back(atoi(classlabel.c_str()));
            }
            /*
        	 * else just omit it
        	 */
        }
    }
    return test_number;
}

static Mat norm_0_255(InputArray _src)
{
	Mat src = _src.getMat();
    /* 
     * Create and return normalized image
     */
    Mat dst;
    switch(src.channels()) {
    case 1:
        cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
        break;
    case 3:
        cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
        break;
    default:
        src.copyTo(dst);
        break;
    }
    return dst;
}
