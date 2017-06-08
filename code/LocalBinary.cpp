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
 * //   下面简要的写一下每个算法的名称以及基本思路
 * 
 * This file is an implementation of Local Binary Pattern .This algorithm
 * concentrated on extracting local features from images.The basic idea of
 * Local Binary Patterns is to summarize the local structure in 
 * an image by comparing each pixel with its neighborhood. 
 * Take a pixel as center and threshold its neighbors against.
 *
 * This file is written by 许泽平(Xu Zeping)
 *
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string> 
#include <vector>
#include <cstdlib>

#include <opencv2/core/core.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/highgui/highgui.hpp>

#define STEP 10

using namespace std;
using namespace cv;

static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }
    string line, path, classlabel;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) {
            images.push_back(imread(path, 0));
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
}

void local_binary(vector<Mat>& read_images, vector<int>& read_labels,int num);

int main(int argc, char *argv[]){
    /*
     * ensure command line argument
     */
    if(argc < 2 || argc > 4){
        cout << "usage: ./" << argv[0]
        << " <csv.ext> <output folder> <number>" << endl;
        return 1;
    }
    
    string path_csv = string(argv[1]);
    string out_folder = "";
    
    if(argc >= 3){
        out_folder = string(argv[2]);
    }
    
    vector<Mat> images;
    vector<int> labels;
    
    try{
        read_csv(path_csv,images,labels);
    }catch(cv::Exception &e){
        cerr << "Opening file error: file\"" << path_csv <<  "\"."
            << endl << "Reason: " << e.msg << endl;
        return 1;
    }
    
    /*
     * ensure the number of images is enough
     */
    if(images.size() <= 2){
        string error_message = "This demo needs at least 3 images \
            to work. Please add more images to your data set!";
        CV_Error(CV_StsError, error_message);
    }
    if(argc == 3)
        local_binary(images,labels,0);
    else
        local_binary(images,labels,atoi(argv[3]));
    
    return 0;
}

/*
 * local_binary - a function to prepare for calling LBPHFaceRecognizer
 *
 */

void local_binary(vector<Mat>& read_images, vector<int>& read_labels,int num){
    /*
     * Get images' height from first picture ,in order to 
     * reshape the images to their true size
     */
    //int height = read_images[0].rows;
    vector<Mat> testImages,images;
    vector<int> testLabels,labels;
    
    if(num == 0){   // the argument of number is defaulted
        num = 2;
    }
    /*
     * Generate the testsample and trainsample
     */
    int n = num;
    for(int i = 0 ; i < read_images.size() / STEP ; ++i){
        int j;
        for(j = i * STEP ; j < i * STEP + num ; ++j){
            testImages.push_back(read_images[j]);
            testLabels.push_back(read_labels[j]);
        }
        for(; j < i * STEP + STEP ; ++j){
            images.push_back(read_images[j]); 
            labels.push_back(read_labels[j]);
        }
    }
    
    /*
     * Function createLBPHFaceRecognizer‘s prototype :
     * Ptr<FaceRecognizer> createLBPHFaceRecognizer(int radius=1,
     * int neighbors=8, int grid_x=8, int grid_y=8, double threshold=DBL_MAX);
     *
     * we can change the argument to create the recognizer we need
     *
     */
    Ptr<FaceRecognizer> model = createLBPHFaceRecognizer(1,8,4,4,123.0);
    model->train(images, labels);
    
    int cor_number = 0;
    int predictlabel;
    cout << "Predict begin: " << endl;
    for(int i = 0 ; i < testImages.size() ; ++i){
        predictlabel = model -> predict(testImages[i]);
        cout << " Predicted class = " << predictlabel << " Actual class = "
        << testLabels[i] << "." ;
        if(predictlabel == testLabels[i]){
            cout << "Correct !" << endl;
            ++cor_number;
        }
        else{
            cout << "False !" << endl;
        }
    }
    cout << "Predict end. " << endl << endl << "Model Information :" << endl;
    cout << " radius: " << model->getInt("radius") << endl << " neighbors: "
    << model->getInt("neighbors") << endl << " grid_x: "<< model->getInt("grid_x")
    << endl << " grid_y: " << model->getInt("grid_y") << endl << " threshold: "
    << model->getDouble("threshold") << endl;
    double cor_rate = (double)cor_number / (double)testImages.size();
    cout << "Result - Correct Rate: " << (int)(cor_rate * 100) << '%'
    << endl;
    
}












