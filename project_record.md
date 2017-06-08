# 算分project——基于OpenCV的人脸识别

小组成员：许泽平		祁俊昆

**OUTLINE：**

[toc]

## step1:配置环境-安装opencv

按照官方网站上的指示完全没法安装 **-_-!!!**

### mac os上的安装过程** 

[linux上的安装方法可以参见这个](http://docs.opencv.org/2.4/doc/tutorials/introduction/linux_install/linux_install.html#linux-installation)
但是只按照官网的教程来操作并不能够直接运行，可以参看[这个博客](http://blog.csdn.net/qccz123456/article/details/52351831)配置相应的环境变量

#### 1.利用homebrew安装相关包
按照官网上的要求，我们需要安装`pkg-config`、`cmake`、`g++\gcc`
在电脑上安装好**homebrew**的情况下，我们可以通过它来安装上述插件，比如要安装cmake时，只需要使用以下命令：
```
brew search cmake
```
根据反馈来看看有没有这个包/名字有没有写错，没有的话则可使用以下命令安装：
```
brew install cmake
```

在安装好上述插件以后，就可以安装opencv了（直接使用homebrew安装比到官网上下载编译要容易）：
```
brew tap homebrew/science
brew install opencv
```

#### 2.设置pkg-config

安装完成以后，在`/usr/local/Cellar`下就会出现一个新的文件夹`opencv`。接下来打开`.bash_profile`配置`pkg-config`的环境变量：
```
open -e .bash_profile
```
在打开的文档中加入：(具体路径要根据实际安装情况而定)
```
PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/Cellar/opencv/2.4.13.2/lib/pkgconfig
export PKG_CONFIG_PATH

export LD_LIBRARY_PATH=/usr/local/Cellar/opencv/2.4.13.2/bin:SLD_LIBRARY_PATH
export PATH=${PATH}:/usr/local/Cellar/opencv/2.4.13.2/lib
```
保存，然后让环境变量生效：
```
source .bash_profile
```
然后输入以下命令：
```
pkg-config --libs opencv
pkg-config --cflags opencv
```
如果显示类似以下，则说明配置已经成功：
```
~ apple$ pkg-config --libs opencv

-L/usr/local/Cellar/opencv/2.4.13.2/lib -lopencv_calib3d -lopencv_contrib -lopencv_core -lopencv_features2d -lopencv_flann -lopencv_gpu -lopencv_highgui -lopencv_imgproc -lopencv_legacy -lopencv_ml -lopencv_nonfree -lopencv_objdetect -lopencv_ocl -lopencv_photo -lopencv_stitching -lopencv_superres -lopencv_ts -lopencv_video -lopencv_videostab

~ apple$ pkg-config --cflags opencv

-I/usr/local/Cellar/opencv/2.4.13.2/include/opencv -I/usr/local/Cellar/opencv/2.4.13.2/include
```

#### 3.测试
以下是一个可供测试用的程序，该程序将同一文件夹内的名为`image.jpg`的文件复制并生成文件`image_copy.png`
```
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;

int main(int argc, char** argv)
{   
    // Load an image from file - change this based on your image name
    Mat img = imread("image.jpg", CV_LOAD_IMAGE_UNCHANGED);

    if(img.empty())
    {
        fprintf(stderr, "failed to load input image\n");
        return -1;
    }

    // this is just to show, that you won't have to pre-alloc
    // result-images with c++ any more..
    Mat gray;
    cvtColor(img,gray,CV_BGR2GRAY);

    // Write the image to a file with a different name,
    // using a different image format -- .png instead of .jpg
    if( ! imwrite("image_copy.png", img) )
    {
        fprintf(stderr, "failed to write image file\n");
    }

    imshow("Display Image", img);
    waitKey(0);

    // no need to release anything with c++ !   
    return 0;
}
```
makefile文件如下：
```
test:test.cpp
	g++ -o test test.cpp `pkg-config --libs opencv` `pkg-config --cflags opencv` 

clean:
	rm test
```
make之后运行即可。


## step2:实验概述

### 0.OpenCV
opencv(open source computer vision)是一个由intel发起的基于bsd许可开源发行的跨平台计算机视觉库。这是一个由c函数和少量c++类构成的库，由于我们的培养方案大多围绕c/c++来展开，所以使用这个库也是比较方便的。
opencv2.4加入的新类`FaceRecognizer`为我们实现了许多人脸识别的算法，本次project的目的就是采用不同的算法来对部分数据库中的图像进行处理，比较不同算法的性能差别，并探究背后的原理。

主要包括的人脸识别算法有：
- 特征脸法(Eigenfaces）
- Fisherfaces
- 局部二进制模式直方图(Local Binary Patterns Histograms) ----这是我自己起的名字

开始我们采用的数据库是[AT&T Facedatabase](http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html)

### 1.准备数据
要运用算法来处理图像，我们需要将所有的图像文件的清单处理成一种特殊的`CSV`格式，便于读取。我们这里的表示将用每一行代表一个文件，前面是一个包含路径的文件名，后面是一个`;`作为分隔符，最后跟着一个标签（一个整数，代表所属人），例如`/path/to/image.ext;0`

我们所采用的数据库的结构以及处理用的python脚本见附录，转换输出的结果如下：
```
/Users/apple/Downloads/orl_faces/s1/1.pgm;0
···
/Users/apple/Downloads/orl_faces/s1/10.pgm;0
```




### 2.特征脸法
特征脸法(Eigenfaces)是基于主成分分析法(Principal Component Analysis，PCA)的一种人脸识别的算法。特征脸法采用整体面部识别的方法:把面部图像看作一个向量，把图像的每一个像素点看作向量的一维。一个$pxq$的图像将被表示成一个$m = pxq$维的向量， (简单地把每一个像素点按照行或者列的方式依次排列而成) 然而如此产生的向量是巨大的，事实上这些高维向量可以被一些本质的、低维的向量所表示，主成分分析法正是用来达到这一目的。 

#### 算法原理: 
主成分分析是一种分析、简化数据集的技术，经常用于减少数据集的维数，同时保持数据集中
的对方差贡献最大的特征。其方法主要是通过对协方差矩阵进行特征分解，以得出数据的主成
分(即特征向量)与它们的权值(即特征值)，其结果可以理解为对原数据中的方差做出解
释:哪一个方向上的数据值对方差的影响最大?直观上，对方差影响越大的方向上对数据的影
响越重要。

#### 算法内容
1、获得包含M张人脸的图像集合S，按照上述方法将每一张人脸图像转化为一个N维向量: 
$S = \{\Gamma_1,\Gamma_2,\Gamma_3,...,\Gamma_4\}$
2.计算平均图像(均值)$ψ$，并计算每张图像与平均图像的差值$Φ_i$: 
$\Psi = \frac{1}{M}\sum\limits_{n = 1}^{M}\Gamma_n$
$\Phi = \Gamma_i - \Psi$
3.求解协方差矩阵C的特征值$λ_i$和单位正交特征向量$u_i$:
$C = \frac{1}{M}\sum\limits_{n = 1}^{M}\Phi_n\Phi_n^\mathrm{T} = AA^\mathrm{T}$ 
$A = \{\Phi_1,\Phi_2,...,\Phi_n\}$
每个特征向量也都是N维向量，转化为图像之后看上去也像是一张“脸”，因而被称为特征脸。通常上我们只取最大的前k个特征值所对应的特征向量作为抓取出的“特征”。 
4、识别人脸。考虑待识别人脸，按照同样的方式转化为N维向量Γ，用特正脸去表征它: 
$\omega_i = u_k^\mathrm{T}(\Psi - \Phi)$
$\omega_i$i表示Γ在特征脸$u_i$下的权值，M个权重可以构成一个向量: 
$\Omega^\mathrm{T} = [\omega_1,\omega_2,...,\omega_n]$
Ω就是Γ在特征脸下的表征，同理S中每张脸都可以用同样的方式表征，计算二者的“差别”， 
通常使用欧氏距离表征这个差别:
$\epsilon_k = \parallel\Omega - \Omega_k\parallel^2$
设定两个阈值ε1和ε2:若εk<ε1，认为Γ和Γk同属一个人;若ε1<εk<ε2，认为Γ也是一张脸，但可能不属于S;若εk>ε2，认为Γ不是一张人脸。 

#### 理论基础
I、为何PCA要寻求协方差矩阵的特征向量? 我们从二维的简单情形入手。
<img src="/Users/apple/Desktop/PCA_1.png" width="35%" /> 
PCA的目的是降维，直观上，希望可以用一维来表示二维的点以降低维度。一维即使一条直线，相当于二维的点向直线上做投影。为了使变换丢失最少的信息，应该让这些点投影后彼此相隔“越远越好”，也就是彼此有区别(否则若两个点重合在一起，就无法区分这两个点，也就是丢失了”信息“)。上图中，直观上认为向u1方向投影应该是一个不错的选择，而向u2这个方向投影就会有很多店重合在一起。 为讨论方便，我们把每个点都减去均值μ，只考虑偏差的部分，这样做也使点分布在原点附近。在这里只用了五个点示意。 
<img src="/Users/apple/Desktop/PCA_2.png" width="35%" /> 
我们先选取一个方向u1，并向这个方向做投影，我们发现各个点之间区别很大，如果向这个方 向降维，应该是不错的选择。 
<img src="/Users/apple/Desktop/PCA_3.png" width="35%" /> 
如果我们换一个方向u2，会发现没有第一次选择效果好。 
<img src="/Users/apple/Desktop/PCA_4.png" width="35%" /> 
那么如何选择向量u，才能使各个点彼此区分，也即上述“使方差最大化”呢? 每个点都可以用一个向量来表示，记为$X_i$，i = 1，2，......，m，选择的降维的方向(向量)记为u。那么$X_i$在u上的投影(原点到投影点的距离)即为$X_i^\mathrm{T}u$，求使方差最大化的方向u也即求u使得
$max \frac{1}{m}\sum\limits_{i=1}^{m}( X_i^\mathrm{T}u)^2 = max \frac{1}{m}\sum\limits_{i=1}^{m} u^\mathrm{T}X_i X_i^\mathrm{T}u = max$ $u^\mathrm{T}(\frac{1}{m}\sum\limits_{i=1}^{m}X_i X_i^\mathrm{T})u$

注意到我们预处理时令每个向量$X_i$都减去了均值μ，因而事实上 
$S = \frac{1}{m}\sum\limits_{i=1}^{m}X_iX_i^\mathrm{T}$ 就是关于X的协方差矩阵

(补充线性代数知识:设实对称矩阵$A_{nxn}$的全部特征值(非负实数)按照从大到小排列为λ1 ≧ λ2 ≧ ...... ≧ λn ≧ 0，对于任意的n维单位向量α，都有**λn ≦ $α^\mathrm{T}A_α$ ≦ λ1**。)

令$\parallel u \parallel$ = 1，即规定u为单位向量，则使上式达到最大值也即求关于X的协方差矩阵S的(单位)特征量u。 
用类似的思路可以将结论拓展到多维情况。我们希望用尽量少的维度代表尽可能全部的信息，因而所选取的k个方向u1、u2、......、uk要保证没有“冗余”信息，也即它们是正交的，并选取最大的前k个特征值所对应的特征向量。 

II、其他一些线性代数知识 
1、实对称矩阵一定可以对角化(有n个线性无关的特征向量)，且特征根都是非负实数。
2、属于不同特征值的特征向量是正交的，属于一个特征值的特征向量可以施密特正交化为正交向量。
3、矩阵$A_{sxn}$和$B_{nxs}$，AB和BA有相同的非零特征值，且特征值的重数相等。设α是AB属于特征值λ的一个特征向量，则Bα是BA属于特征值λ的一个特征向量:ABα = λα => B(ABα) = B(λα) => BA(Bα) = λ(Bα)。
利用上定理，可以简化上述第3部中的计算，当M << N时，可以显著的简化运算和存储 (N x N的矩阵与M x M的矩阵)。 
4、若选取特征向量时，只取最大的前k个特征值所对应的特征向量，分别记为u1、u2、......、uk，令W =(u1，u2，......，uk)，记$P = W^\mathrm{T}A$，其中$A =(Φ_1，Φ_2，......， Φ_M)$，记$P =(v_1，v_2，......，v_M)$，则$v_k = W^\mathrm{T}Φ_k$，可见P是A降维后的向量组，这也正是 PCA降维的体现。因为我们取的u1、u2、......、uk是单位正交矩阵，故有$WW^\mathrm{T} = I$，从而$Φ_k = Wv_k$，也即我们可以从低维的$v_k$得到高维的$Φ_k$。 

#### 代码实现
```
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
        Ptr<FaceRecognizer> model = createEigenFaceRecognizer();
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
```


### 3.Fisherfaces
主成分分析（PCA）是Eigenfaces方法的核心，它找到最大化数据总方差的特征的线性组合。虽然这是一种表达数据的强大方法，然而它并没有考虑到所有的情况。所以当抛出组件时，很多变化的信息可能会丢失。想象一下，数据的变化是由外部来源产生的，在这里我们假设这样的变化的来源是光，由PCA识别的组件并不需要包含任何变化信息，所以样本们会被混合到一起，且无法再进行分类。
线性判别分析最大化了类之间到类内部的分散，而不是最大化整体的分散。这个想法很简单：相同的类应该集中在一起，而不同的类在低维表示中尽可能远离彼此。

#### 算法原理
**我们先来考虑数据集中只有两类的情况**
实际上这个算法要处理的情况远不止两类，但为了简便起见，我们先从两类的情况开始讨论。
对于每一张图，我们用一个点在二维坐标系中的坐标来刻画这张图的性质。那么现在的数据是散布在平面上的二维数据，现在想用一维的量来刻画这些数据并将它们区分开来，那么一个合理的想法就是去寻找一个平面内合适的向量并计算出这些点在这个平面上的投影，用数学公式来表示的话就是 $y = w^\mathrm{T} x $ 
<img src="/Users/apple/Desktop/FISHER_1.png" width="75%" /> 

这里借用了一位网友的图（见参考2）。该图中给出了两种可能的方案，直观上第二种方法更好，能够有效的区分开这两类数据集。为了让计算机来实现这一想法，必须要有一种确定的算法来确定这个向量$w$

**下面讨论数据集中有更多类的情况**
我们假设数据共有c类。
一开始我们将图像按照像素逐行逐列的处理成一个向量，这样的话一个100x100的图像得到的向量就是10000维，设其为$x$，其维数为n。
如果这种时候还只采用一个向量来区分的话有可能会无效，例如

<img src="/Users/apple/Desktop/FISHER_2.png" width="50%" /> 
出现了这样的某个向量恰好成为某几个点组成平面的法向量的情况。
这种时候一个可靠的解决方案就是采用多个向量（采用矩阵表示）


#### 公式推导
设$W$为
$W = [w_1 | w_2 |...| w_K]$
其中$w_i$是一个n维列向量
那么$x$的投影就可以表示为
$y = w^\mathrm{T} x$
其中，$y$是一个k维向量。

这之后从类间散列度和类内散列度来考虑。

首先计算每类数据的均值(中心点)：
$\mu_i = \frac{1}{N_i}\sum\limits_{x\in\omega_i}x$
这里的下标i代表了这一类，$N_i$表示这一类中的元素个数。所以$\mu_i$就代表这一类的中心。

整个样本的中心定义为：
$\mu = \frac{1}{N}\sum\limits_{\forall x}x = \frac{1}{N}\sum\limits_{\forall x}N_i\mu_i$

变量类中散列度$S_w$定义如下：
$S_w = \sum\limits_{i=1}^{c}S_{wi}$
其中：$S_{wi} = \sum\limits_{x\in\omega_i}(x-\mu_i)(x-\mu_i)^\mathrm{T}$   代表第i类内的散列度，是一个nxn矩阵（个人理解：这个公式实际上和方差公示一样，这样做运算也是将各个维度和它们之间的相互作用放在了一个平等的位置上来看待）

类间散列度$S_B$定义如下:
$S_B = \sum\limits_{i=1}^{c}N_i(\mu_i-\mu)(\mu_i-\mu)^\mathrm{T}$
表示各个类别到样本中心的距离，也是一个nxn矩阵，其中的$N_i$代表一个类别中的样本数，也就是这个人的图片个数

以上都是投影之前的数据。下面我们来计算投影后的数据：

投影后第i类的样本中心：
$\widetilde{\mu_i} = \frac{1}{N_i}\sum\limits_{y\in\omega_i}y$

投影后的总样本中心
$\widetilde{\mu} = \frac{1}{N}\sum\limits_{\forall y}y$

投影后的类中散列度：
$\widetilde{S_w} = \sum\limits_{i=1}^{c}\sum\limits_{y\in\omega_i}(y - \widetilde{\mu_i})(y - \widetilde{\mu_i})^\mathrm{T}$

投影后的类间散列度：
$\widetilde{S_B} = \sum\limits_{i=1}^{c}N_i(\widetilde{\mu_i}-\widetilde{\mu})(\widetilde{\mu_i}-\widetilde{\mu})^\mathrm{T}$

根据以上公式以及线性代数知识，有：
$\widetilde{S_w} = W^\mathrm{T} S_w W$
$\widetilde{S_B} = W^\mathrm{T} S_B W$

如何判断W是不是最佳呢，可以从两方面考虑：1、不同的分类得到的投影点要尽量分开（$S_B$尽量大）；2、同一个分类投影后得到的点要尽量聚合（$S_w$尽量小）

由此我们可以定义度量用的参数
$J(W) = \frac{|\widetilde{S_B}|}{|\widetilde{S_w}|} = \frac{|W^\mathrm{T} S_B W|}{|W^\mathrm{T} S_B W|}$

上式取极大值时的W的选取是一个组合优化问题，被证明满足以下式子：
$S_w^{-1}S_B w_i = \lambda w_i$
即：$w_i$是矩阵$S_w^{-1}S_B$的特征值为$\lambda$的特征向量，这里根据需求，只选取前k大个特征值的特征向量

计算出这些之后，我们就可以回到之前的问题，得出了对应的投影向量，我们就可以依据它们来对向量进行分类了。得到了k个特征向量，如何匹配某人脸和数据库内人脸是否相似呢，方法是将这个人脸在k个特征向量上做投影，得到k维的列向量或者行向量，然后和已有的投影求得欧式距离，根据阈值来判断是否匹配。

另外还需注意：
由于$S_B$中的（μi-μ）秩为1，所以$S_B$的至多为C（矩阵的秩小于等于各个相加矩阵的和）。又因为知道了前C-1个μi后，最后一个μc可以用前面的μi来线性表示，因此$S_B$的秩至多为C-1，所以矩阵的特征向量个数至多为C-1。因为C是数据集的类别，所以假设有N个人的照片，那么至多可以取到N-1个特征向量来表征原数据。（存疑）

#### 代码实现
```
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
```



### 4.Local Binary Patterns Histograms
与之前提到的算法不同，LBP算法是通过提取局部特征作为判断依据的。LBP方法的显著优点是对光照不敏感，但是仍然没有解决表情与姿态的问题。不过相比于特征脸方法，识别率已经有了极大的提升。
特征面和Fisherfaces采取了一种整体的方法来进行面部识别。算法将数据视为高维图像空间中的某个矢量。我们都知道高维度不好，所以确定了一个较低维的子空间，其中（可能）有用的信息被保留。特征面方法最大化总散射，如果方差是由外部源产生的，则可能导致问题，因为在所有类别上具有最大方差的分量不一定对分类有用。因此，为了保留一些区别用的信息，我们应用了线性判别分析（LDA），并按照Fisherfaces方法的描述进行了优化。 Fisherfaces方法至少对于我们在模型中假设的约束场景运行良好。
但现实生活并不完美。我们根本不能在同一个人的10种不同的图像中保证完美的光线设置。那么如果每个人只有一个图像呢？我们对子空间的协方差估计可能是非常错误的。而我们实际需要多少幅图像来获得有用的估计值？就算在AT&T图像库中，对于前两种算法想要得到一个很好的识别率，我们对同一个人至少需要8张照片，并且此时Fisherfaces算法并没有什么提升。
后来，大部分研究集中在从图像中提取局部特征。这样的算法不是把整个图像看成一个高维向量，而只是一个对象的局部属性。可一个事物的图像表示会受到照明变化、图像的旋转等因素的影响。所以我们必须要求算法对这些东西具有一定的健壮性（robust）。算法的基本思想是通过将每个像素与其邻域进行比较来总结图像中的局部结构。以像素为中心，并对其邻居进行阈值。如果中心像素的强度大于等于其邻居，则表示为1，如果不是则为0。

#### 算法思想
最初的LBP是定义在像素3x3的邻域内的，以领域中心点的像素为阈值，将相邻的8个像素的灰度值与其进行比较，若周围像素值大于中心像素值，则该像素点的位置被标记为1，否则为0。这样，3x3邻域内的8个点经比较可产生8位二进制数（通常转换为十进制数即LBP码，共256种），即得到该邻域中心像素点的LBP值，并用这个值来反映该区域的纹理信息。如下图所示：
<img src="/Users/apple/Desktop/LBP_1.png" width="75%" /> 

写成数学公式的话是：
$LBP(x_c,y_c) = \sum\limits_{p=0}^{P-1}2^ps(i_p - i_c)$
其中，$(x_c,y_c)$表示中心元素，$i_c$是中心元素像素值

#### 算法改进

##### 1.圆形LBP算子
基本的 LBP算子的最大缺陷在于它只覆盖了一个固定半径范围内的小区域，这显然不能满足不同尺寸和频率纹理的需要。

为了适应不同尺度的纹理特征，并达到灰度和旋转不变性的要求，Ojala等对 LBP 算子进行了改进，将 3×3邻域扩展到任意邻域，并用圆形邻域代替了正方形邻域，改进后的 LBP 算子允许在半径为 R 的圆形邻域内有任意多个像素点。从而得到了**诸如半径为R的圆形区域内含有P个采样点的LBP算子**。比如下图定了一个5x5的邻域：
<img src="/Users/apple/Desktop/LBP_2.png" width="30%" /> 

该图中的八个采样点可以使用如下公式计算：
$x_p = x_c + Rcos(\frac{2\pi p}{P})$
$y_p = y_c - Rsin(\frac{2\pi p}{P})$   (减号是因为我们通常按照顺时针来计算)

不过我们必须考虑到一点，如果按照这个公式计算出的值不是整数，那么这个位置所对应的像素不存在。这个时候我们必须采用某种方法得到一个最恰当的像素位置。图像处理领域常常采用一种被称为**双线性插值**的方式来解决这一问题。

$f(x,y) \approx \begin{bmatrix}1-x & x\end{bmatrix} \begin{bmatrix}f(0,0) & f(0,1) \\ f(1,0) & f(1,1) \end{bmatrix} \begin{bmatrix} 1-y \\ y \end{bmatrix}$

注：关于双线性插值公式的含义
这里的f函数所表示的即是由坐标得到灰度值的函数。该公式的意思是，选取适当的坐标系，使得我们所得到的函数点的坐标周围正方形的四个点的坐标分别为$(0,0)、(0,1)、(1,0)、(1,1)$，在此坐标系下通过上式计算得到的值就可视为该点灰度值的一个估计。

##### 2.LBP等价模式
其实最重要的部分在上面便已经介绍完了。不过这里稍微提一下一种优化形式。
之前的算子所能得到的LBP值与其采取的点数是指数关系。为了降低这样的复杂度，产生了一种LBP等价模式算法来对LBP算子的模式种类来进行降维。在实际图像中，绝大多数LBP模式最多只包含两次从1到0或从0到1的跳变。因此，Ojala将“等价模式”定义为：当某个LBP所对应的循环二进制数从0到1或从1到0最多有两次跳变时，该LBP所对应的二进制就称为一个等价模式类。如00000000（0次跳变），00000111（只含一次从0到1的跳变），10001111（先由1跳到0，再由0跳到1，共两次跳变）都是等价模式类。除等价模式类以外的模式都归为另一类，称为混合模式类，例如10010111（共四次跳变）。通过这样的改进，二进制模式的种类大大减少，而不会丢失任何信息。模式数量由原来的$2^p$种减少为 $P(P-1)+2$种，其中P表示邻域集内的采样点数。对于3×3邻域内8个采样点来说，二进制模式由原始的256种减少为58种，这使得特征向量的维数更少，并且可以减少高频噪声带来的影响。

#### 代码框架
```
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
    Ptr<FaceRecognizer> model = createLBPHFaceRecognizer(1,16,4,4,123.0);
    model->train(images, labels);
    
    int cor_number = 0;
    int predictlabel;
    cout << "Predict begin: " << endl;
    for(int i = 0 ; i < testImages.size() ; ++i){
        predictlabel = model -> predict(testImages[i]);
        cout << " Predicted class = " << predictlabel << "/ Actual class = "
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
```


## step3:得出实验结果
这里我们最初使用了前文中提到过的AT&T数据库得出各个算法测试准确度的状态
![](/Users/apple/Desktop/result_1.png)

下面是具体的正确率数值。

![](/Users/apple/Desktop/result_2.png)

在这个数据库下，几个算法的性能存在着相对的差异，我们可以得到以下结论：

- 性能上，Local binary pattern histogram 优于eigenfaces，eigenfaces又要优于fisherfaces(这一点是因为fisherface优化的方面并不能在这里充分的体现)
- local binary pattern histogram能达到满的准确率，而另外两个算法没能达到（事实上，在识别条件，尤其是光照因素变化比较剧烈时，该算法的表现会远优于另外两个算法）

## step4:应用——性别识别
以上算法除了能够用来进行人脸的匹配外，还可以有许多其他的应用，比如用来做性别识别。实现的原理其实相当的简单。

性别识别无非就是把训练样本的标签变成只有两类：男性和女性。需要说明的是，EigenFace是基于PCA的，是一种非监督的模型，不太适合性别识别的任务。而正如我们前面所讨论过的一样，Fisherfaces方法除了考虑类内的关系，还考虑到了其间的相互作用。所以用来做性别识别性能是优于eigenface的。

这部分的代码略去（实际上可以直接复用之前的代码）



## Appendix附录
### 1.数据库结构

```
orl_faces apple$ tree
.
├── README
├── record.md
├── s1
│   ├── 1.pgm
│   ···
│   └── 10.pgm
├── s10
│   ├── 1.pgm
│   ···
│   └── 10.pgm
├── s11
│   ├── 1.pgm
│   ···
│   └── 10.pgm
├── s12
│   ├── 1.pgm
│   ···
│   └── 10.pgm
···
├── s19
│   ├── 1.pgm
│   ···
│   └── 10.pgm
├── s2
│   ├── 1.pgm
│   ···
│   └── 10.pgm
├── s20
···
├── s40
│   ├── 1.pgm
│   ···
│   └── 10.pgm
├── s5
│   ├── 1.pgm
│   ···
│   └── 10.pgm
├── s6
│   ├── 1.pgm
│   ···
│   └── 10.pgm
├── s7
│   ├── 1.pgm
│   ···
│   └── 10.pgm
├── s8
│   ├── 1.pgm
│   ···
│   └── 10.pgm
└── s9
    ├── 1.pgm
    ···
    └── 10.pgm
```

### 2.转换为CSV采用的python脚本
如下：
```
#!/usr/bin/env python

import sys
import os.path

# This is a tiny script to help you creating a CSV file from a face
# database with a similar hierarchie:
#
#  philipp@mango:~/facerec/data/at$ tree
#  .
#  |-- README
#  |-- s1
#  |   |-- 1.pgm
#  |   |-- ...
#  |   |-- 10.pgm
#  |-- s2
#  |   |-- 1.pgm
#  |   |-- ...
#  |   |-- 10.pgm
#  ...
#  |-- s40
#  |   |-- 1.pgm
#  |   |-- ...
#  |   |-- 10.pgm
#

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print "usage: create_csv <base_path>"
        sys.exit(1)

    BASE_PATH=sys.argv[1]
    SEPARATOR=";"

    label = 0
    for dirname, dirnames, filenames in os.walk(BASE_PATH):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                abs_path = "%s/%s" % (subject_path, filename)
                print "%s%s%d" % (abs_path, SEPARATOR, label)
            label = label + 1
```

## Reference参考
1.[opencv官方文档](http://docs.opencv.org/2.4/modules/contrib/doc/facerec/index.html)
2.[人脸识别经典算法三：Fisherface（LDA）](http://blog.csdn.net/smartempire/article/details/23377385)
3.[人脸识别经典算法二：LBP方法](http://blog.csdn.net/smartempire/article/details/23249517)
4.[人脸识别经典算法一：特征脸方法（Eigenface）](http://blog.csdn.net/smartempire/article/details/21406005)

