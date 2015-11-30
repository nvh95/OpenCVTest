#include <bits/stdc++.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>

using namespace std;
using namespace cv;
using namespace cv::ml;

bool plotSupportVectors = true;
int numTrainingPoints = 20000;
int numTestPoints = 2000;
int size = 200;
int eq = 0;

// accuracy
float evaluate(cv::Mat& predicted, cv::Mat& actual) {
    assert(predicted.rows == actual.rows);
    int t = 0;
    int f = 0;
    for(int i = 0; i < actual.rows; i++) {
        float p = predicted.at<float>(i,0);
        float a = actual.at<float>(i,0);
        if((p >= 0.0 && a >= 0.0) || (p <= 0.0 &&  a <= 0.0)) {
            t++;
        } else {
            f++;
        }
    }
    return (t * 1.0) / (t + f);
}

// plot data and class
void plot_binary(cv::Mat& data, cv::Mat& classes, string name) {
    cv::Mat plot(size, size, CV_8UC3);
    plot.setTo(cv::Scalar(255.0,255.0,255.0));
    for(int i = 0; i < data.rows; i++) {

        float x = data.at<float>(i,0) * size;
        float y = data.at<float>(i,1) * size;

        if(classes.at<float>(i, 0) > 0) {
            cv::circle(plot, Point(x,y), 2, CV_RGB(255,0,0),1);
        } else {
            cv::circle(plot, Point(x,y), 2, CV_RGB(0,255,0),1);
        }
    }
    cv::imshow(name, plot);
}

//// function to learn
int f(float x, float y, int equation) {
    switch(equation) {
    case 0:
        return y > sin(x*10) ? -1 : 1;
        break;
    case 1:
        return y > cos(x * 10) ? -1 : 1;
        break;
    case 2:
        return y > 2*x ? -1 : 1;
        break;
    case 3:
        return y > tan(x*10) ? -1 : 1;
        break;
    default:
        return y > cos(x*10) ? -1 : 1;
    }
}

// label data with equation
cv::Mat labelData(cv::Mat points, int equation) {
    cv::Mat labels(points.rows, 1, CV_32SC1);
    for(int i = 0; i < points.rows; i++) {
             float x = points.at<float>(i,0);
             float y = points.at<float>(i,1);
             labels.at<float>(i, 0) = f(x, y, equation);
        }
    return labels;
}

void svm(cv::Mat& trainingData, cv::Mat& trainingClasses, cv::Mat& testData, cv::Mat& testClasses) {
    //CvSVMParams param = CvSVMParams();
    Ptr<ml::SVM> param = ml::SVM::create();

    param->setType(ml::SVM::C_SVC);

    param->setKernel(ml::SVM::RBF); //CvSVM::RBF, CvSVM::LINEAR ...
    param->setDegree(0); // for poly
    param->setGamma(20); // for poly/rbf/sigmoid
    param->setCoef0(0); // for poly/sigmoid

    param->setC(7); // for CV_SVM_C_SVC, CV_SVM_EPS_SVR and CV_SVM_NU_SVR
    param->setNu(0.0); // for CV_SVM_NU_SVC, CV_SVM_ONE_CLASS, and CV_SVM_NU_SVR
    param->setP(0.0); // for CV_SVM_EPS_SVR

    //param->setClassWeights(NULL); // for CV_SVM_C_SVC
//    param->setTermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 1000, 1e-6);

    // SVM training (use train auto for OpenCV>=2.0)
    param->train(trainingData, ml::ROW_SAMPLE, trainingClasses);

    cv::Mat predicted(testClasses.rows, 1, CV_32F);

    for(int i = 0; i < testData.rows; i++) {
        cv::Mat sample = testData.row(i);

        float x = sample.at<float>(0,0);
        float y = sample.at<float>(0,1);

        predicted.at<float>(i, 0) = param->predict(sample);
    }

    cout << "Accuracy_{SVM} = " << evaluate(predicted, testClasses) << endl;
    plot_binary(testData, predicted, "Predictions SVM");

    // plot support vectors
    if(plotSupportVectors) {
        cv::Mat plot_sv(size, size, CV_8UC3);
        plot_sv.setTo(cv::Scalar(255.0,255.0,255.0));

        Mat svec = param->getSupportVectors();
        for(int vecNum = 0; vecNum < svec.rows; vecNum++) {
            const float* vec = svec.ptr<float>(vecNum);
            cv::circle(plot_sv, Point(vec[0]*size, vec[1]*size), 3 , CV_RGB(0, 0, 0));
        }

   // imwrite("result.png", plot_sv);        // save the image
    cv::imshow("Support Vectors", plot_sv);
    }
}

void mlp(cv::Mat& trainingData, cv::Mat& trainingClasses, cv::Mat& testData, cv::Mat& testClasses) {

    cv::Mat layers = cv::Mat(4, 1, CV_32SC1);

    layers.row(0) = cv::Scalar(2);
    layers.row(1) = cv::Scalar(10);
    layers.row(2) = cv::Scalar(15);
    layers.row(3) = cv::Scalar(1);

    Ptr< ANN_MLP > params =  ml::ANN_MLP::create();
//    CvANN_MLP_TrainParams params;
//    CvTermCriteria criteria;

    params->setLayerSizes(layers);
    params->setActivationFunction( ANN_MLP::SIGMOID_SYM);
    params->setTrainMethod(ANN_MLP::BACKPROP);
    params->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 100, 0.00001f));

    params->setBackpropMomentumScale(0.05f);
    params->setBackpropWeightScale(0.05f);

//    // train
    Mat trainingResponses = Mat::zeros( trainingData.rows, 1, CV_32F );
    for (int i=0; i<trainingClasses.rows; i++) {
        trainingResponses.at<float>(i, 0) = trainingClasses.at<float>(i, 0)*1.0f;
       // cout << trainingResponses.at<float>(i, 0) << endl;
    }

    Ptr<TrainData> tdata = TrainData::create(trainingData, ROW_SAMPLE, trainingResponses);

    //params->train(tdata);
    params->train(tdata);

    cv::Mat response(1, 1, CV_32FC1);
    cv::Mat predicted(testClasses.rows, 1, CV_32F);
    for(int i = 0; i < testData.rows; i++) {
        cv::Mat response(1, 1, CV_32FC1);
        cv::Mat sample = testData.row(i);

        params->predict(sample, response);
        predicted.at<float>(i,0) = response.at<float>(0,0);

    }

    cout << "Accuracy_{MLP} = " << evaluate(predicted, testClasses) << endl;
    plot_binary(testData, predicted, "Predictions Backpropagation");
}

void knn(cv::Mat& trainingData, cv::Mat& trainingClasses, cv::Mat& testData, cv::Mat& testClasses, int K) {

    Ptr<KNearest> knn = KNearest::create();
    knn->setDefaultK(K);
    knn->setIsClassifier(false);
    knn->train(trainingData, ROW_SAMPLE, trainingClasses);

    cv::Mat predicted(testClasses.rows, 1, CV_32F);
    for(int i = 0; i < testData.rows; i++) {
            const cv::Mat sample = testData.row(i);
            predicted.at<float>(i,0) = knn->findNearest(sample, K, noArray());
    }

    cout << "Accuracy_{KNN} = " << evaluate(predicted, testClasses) << endl;
    plot_binary(testData, predicted, "Predictions KNN");
}

void bayes(cv::Mat& trainingData, cv::Mat& trainingClasses, cv::Mat& testData, cv::Mat& testClasses) {

    Ptr< NormalBayesClassifier > bayes = NormalBayesClassifier::create();
    bayes->train(trainingData, ROW_SAMPLE, trainingClasses);
    cv::Mat predicted(testClasses.rows, 1, CV_32F);
    for (int i = 0; i < testData.rows; i++) {
        const cv::Mat sample = testData.row(i);
        predicted.at<float> (i, 0) = bayes->predictProb(sample, noArray(), noArray());
    }

    cout << "Accuracy_{BAYES} = " << evaluate(predicted, testClasses) << endl;
    plot_binary(testData, predicted, "Predictions Bayes");
}

void decisiontree(cv::Mat& trainingData, cv::Mat& trainingClasses, cv::Mat& testData, cv::Mat& testClasses) {

    Ptr< DTrees >  dtree = DTrees::create();
    cv::Mat var_type(3, 1, CV_8U);

    // define attributes as numerical
    var_type.at<unsigned int>(0,0) = VAR_NUMERICAL;
    var_type.at<unsigned int>(0,1) = VAR_NUMERICAL;
    // define output node as numerical
    var_type.at<unsigned int>(0,2) = VAR_NUMERICAL;
    Ptr<TrainData> tdata = TrainData::create(trainingData, ROW_SAMPLE, trainingClasses, noArray(), noArray(), noArray(), var_type);
    dtree->setMaxDepth(11);
    dtree->setMinSampleCount(10);
    dtree->setMaxCategories(15);

    dtree->train(trainingData, ROW_SAMPLE, trainingClasses);
    cv::Mat predicted(testClasses.rows, 1, CV_32F);
    for (int i = 0; i < testData.rows; i++) {
        const cv::Mat sample = testData.row(i);
        //DTrees::Node* prediction = dtree->predict(sample);
        predicted.at<float> (i, 0) = dtree->predict(sample);
    }

    cout << "Accuracy_{TREE} = " << evaluate(predicted, testClasses) << endl;
    plot_binary(testData, predicted, "Predictions tree");
}


int main() {

    cv::Mat trainingData(numTrainingPoints, 2, CV_32FC1);
    cv::Mat testData(numTestPoints, 2, CV_32FC1);

    cv::randu(trainingData,0,1);
    cv::randu(testData,0,1);
//
//    for (int i=0; i<numTrainingPoints; i++)
//        cout << trainingData.cols << endl;

    cv::Mat trainingClasses = labelData(trainingData, eq);
    cv::Mat testClasses = labelData(testData, eq);

    plot_binary(trainingData, trainingClasses, "Training Data");
    plot_binary(testData, testClasses, "Test Data");

    svm(trainingData, trainingClasses, testData, testClasses);
    mlp(trainingData, trainingClasses, testData, testClasses);
    knn(trainingData, trainingClasses, testData, testClasses, 3);
    bayes(trainingData, trainingClasses, testData, testClasses);
  //  decisiontree(trainingData, trainingClasses, testData, testClasses);

    cv::waitKey(0);

    return 0;
}



