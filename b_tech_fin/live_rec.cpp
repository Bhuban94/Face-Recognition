#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>

#include "opencv2/core/core.hpp"
#include "opencv2/core/core_c.h"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdlib.h>

using namespace cv;
using namespace std;
static Mat norm_0_255(InputArray _src) {
	Mat src = _src.getMat();
	Mat dst;
	switch (src.channels()) {
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
static void read_csv(const string& filename, std::vector<Mat>& images, std::vector<int>& labels, char separator = ';') {
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
		if (!path.empty() && !classlabel.empty()) {
			Mat temp = imread(path, 0);
			cout << path << endl;
			images.push_back(imread(path, 0));
			labels.push_back(atoi(classlabel.c_str()));
		}
	}
}
int main() {

	try
	{
		dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
		dlib::shape_predictor sp;
		dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> sp;
		cout << "hello\n";

		string output_folder = ".";
		string tfb_csv = "list.txt";
		//string sfb_csv = "sfb_csv.txt";
		std::vector<Mat> images;
		std::vector<int> labels;
		std::vector<Mat> test_images;
		std::vector<int> test_labels;
		try {
			read_csv(tfb_csv, images, labels);
			//read_csv(sfb_csv, test_images, test_labels);
		}
		catch (cv::Exception& e) {
			cerr << "Error opening file \"" << tfb_csv << "\". Reason: " << e.msg << endl;
			exit(1);
		}
		if (images.size() <= 1) {
			string error_message = "Atleast 2 images needed in database!";
			CV_Error(CV_StsError, error_message);
		}
		cout << images.size() << " " << test_images.size() << endl;;
		Ptr<FaceRecognizer> model = createFisherFaceRecognizer();
		//Ptr<FaceRecognizer> model = createEigenFaceRecognizer();
		model->train(images, labels);
		cout << "training done." << endl;
		model->save("model.xml");
		cout << "saving done." << endl;
		//int predictedLabel = model->predict(testSample);
		CvCapture* cap;
		cap = cvCaptureFromCAM(0);
		Mat frame;
		Mat frame_gray;
		Mat frame_hist;
		Mat background;
		int c = 0;;
		if (cap)
		{
			dlib::image_window win,win_faces;
			while (true){
				frame = cvQueryFrame(cap);
				cvtColor(frame, frame_gray, CV_BGR2GRAY);
				dlib::array2d<dlib::rgb_pixel> img;
				assign_image(img, dlib::cv_image<dlib::bgr_pixel>(frame));
				if (!frame.empty())
				{
					std::vector<dlib::rectangle> dets = detector(img);
					cout << "Number of faces detected: " << dets.size() << endl;
					std::vector<dlib::full_object_detection> shapes;
					for (unsigned long j = 0; j < dets.size(); ++j)
					{
						dlib::full_object_detection shape = sp(img, dets[j]);
						shapes.push_back(shape);
					}
					dlib::array<dlib::array2d<dlib::rgb_pixel> > face_chips;
					extract_image_chips(img, get_face_chip_details(shapes), face_chips);
					int count = 0;
					dlib::array2d<dlib::rgb_pixel> chip;
					for (count; count< face_chips.size(); ++count)
					{
						extract_image_chip(face_chips[count], dlib::rectangle(20, 20, 179, 179), chip);
						Mat face = dlib::toMat(chip);
						cvtColor(face, face, CV_BGR2GRAY);
						equalizeHist(face, face);
						int prediction;
						double confidence;
						model->predict(face, prediction, confidence);
						cout << "Prediction:" << prediction << endl;
					}
					/*
					vector<Rect> faces;
					cvtColor(frame, frame_gray, CV_BGR2GRAY);
					equalizeHist(frame_gray, frame_hist);
					face_cascade.detectMultiScale(frame_hist, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(70, 70));
					for (int i = 0; i < faces.size(); i++)
					{
					rectangle(frame, faces[i], CV_RGB(0, 255, 0), 1);
					Mat face_resized;
					resize(frame_gray(faces[i]), face_resized, Size(98, 112), 1.0, 1.0, INTER_CUBIC);
					//equalizeHist(face_resized, face_resized);
					Mat face_ellipse = face_resized.clone();
					absdiff(face_resized, face_ellipse, face_ellipse);
					ellipse(face_ellipse, Point(49, 60), Size(40.0, 55.0), 0, 0, 360, Scalar(255, 255, 255), -1, 8);
					bitwise_and(face_resized, face_ellipse, face_resized);
					imshow("masked", face_resized);
					int prediction;
					double confidence;
					model->predict(face_resized, prediction, confidence);
					string box_text = format("Prediction = %d, Confidence= %d", prediction, confidence);
					int pos_x = max(faces[i].tl().x - 10, 0);
					int pos_y = max(faces[i].tl().y - 10, 0);
					putText(frame, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 255, 0), 2.0);
					}
					int c = waitKey(10);
					if ((char)c == 'b') { background = frame.clone(); cvtColor(background, background, CV_BGR2GRAY); }
					else if ((char)c == 'c') { break; }
					imshow("video_feed", frame_gray);*/
					win.set_image(img);
					win_faces.set_image(tile_images(face_chips));
				}
				else
				{
					cout << "frame empty\n";
				}
				/*if (!background.empty()){
					vector<Rect> faces;
					Mat difference;
					Mat mask;
					absdiff(frame_gray, background, difference);
					threshold(difference, mask, 20, 255, 0);
					bitwise_and(frame_hist, mask, mask);
					Mat dif = mask.clone();
					imshow("difference", dif);
					face_cascade.detectMultiScale(frame_hist, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(70, 70));
					cvtColor(mask, mask, CV_GRAY2BGR);
					for (int i = 0; i < faces.size(); i++)
					{
					rectangle(mask, faces[i], CV_RGB(0, 0, 255), 1);
					}
					imshow("background", mask);

					}*/

			}
		}
		/*for (int i = 0; i < test_labels.size(); ++i)
		{
		string result_message = format("Predicted class = %d / Actual class = %d.", model->predict(test_images[i]), test_labels[i]);
		//imshow("image" + char(i + 49),test_images[i]);
		cout << result_message << endl;
		}*/
		system("pause");
		waitKey(0);
		return 0;
	}
	catch (exception &e)
	{
		cout << "\nexception thrown!" << endl;
		cout << e.what() << endl;
	}
}