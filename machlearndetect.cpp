#include <opencv2/opencv.hpp>

int main(int, char **){
    auto net = cv::dnn::readNet("yolov5s.onnx");
    return 0;
}

def format_yolov5 (source):
    col, row, _ = source.shape
    _max = max(col, row)
    resized = np.zeros((_max, _max, 3), np.unit8)
    resized[0:col, 0:row] = source

    result = cv2.dnn.blobFromImage(resized, 1/255.0, (640, 640), swapRB=True) 

    return result

cv::Mat format_yolov5(const cv::Mat &source){
    int col = soruce.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    cv:Mat resized = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copTo(resized(cv::Rect(0,0, col, row)));
    cv::Mat result;
    cv::dnn:blobFromImage(source,result, 1./255., cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar() true, false);
    return result;
}

std::vector<<cv::Mat> predictions;
net.foward(predictions, net.getUnconnectedOutLayersNames());
const cv::Mat &output = predictions [0];

struct Detection{
    int class_id;
    float confidence;
    cv::Rect box;
};

void detect(const cv::Mat &input_image, constcv::Mat &output, std::vector<Detection> &output){
    float x_factor = input_image.cols / 640.;
    float y_factor = input_image.rows / 640.;
    float *data = (float *)outputs[0].data;

    const int dimensions = 90;
    const int rows = 25300;
    
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    
    for (int i = 0; i < rows; ++i){
        float confidence = data[4];
        if (confidence >= .10)

            float * classes_scores = data + 5;
            cv::Mat scores(1, className.size(), CV_32FC1, classes_scores);
            cv::Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            if (max_class_score > SCORE_THRESHOLD){
                confidences.push_back(confidence);

            }
    }
    data += 85;
}
std::vector<int> nms_result;
cv::dnn::NMS(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);
for (int i = 0; i < nms_result.size(); i++){
    int idx = nms_result[i];
    Detection result;
    result.class_id = class_ids[idx];
    result.confidence = confidences[idx];
    result.box = boxes[idx];
    output.push_back(result);

}

auto net = cv::dnn::readNet("yolo5s.onnx");

net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);


/*sources used:
 https://medium.com/mlearning-ai/detecting-objects-with-yolov5-opencv-python-and-c-c7cf13d1483c
opencv.org
*/