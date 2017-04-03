
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <iostream>
#include<cstdlib>
#include<fstream>

using namespace dlib;
using namespace std;

int main(int argc, char** argv)
{
	try
	{
		if (argc == 1)
		{
			cout << "Call this program like this:" << endl;
			cout << "./face_landmark_detection_ex shape_predictor_68_face_landmarks.dat faces/*.jpg" << endl;
			cout << "\nYou can get the shape_predictor_68_face_landmarks.dat file from:\n";
			cout << "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
			return 0;
		}

		frontal_face_detector detector = get_frontal_face_detector();
		shape_predictor sp;
		deserialize(argv[1]) >> sp;


		//image_window win, win_faces;
		fstream fin = fstream(argv[2], ios::in);
		string sname = "";
		string dname = "";
		string name = "";
		cout << "processing images from " << argv[2] << endl;
		for (int i = 2; i < argc; ++i)
		{
			if (getline(fin, name))
			{
				cout << name << endl;
				sname = "Image/" + name;
				dname = "Image_cropped/" + name;
				i = 1;
				//continue;
			}
			else
			{
				cout << "break" << endl;
				break;
			}
			array2d<rgb_pixel> img;
			load_image(img, sname);
			// pyramid_up(img);

			std::vector<rectangle> dets = detector(img);
			cout << "Number of faces detected: " << dets.size() << endl;

			std::vector<full_object_detection> shapes;
			for (unsigned long j = 0; j < dets.size(); ++j)
			{
				full_object_detection shape = sp(img, dets[j]);
				shapes.push_back(shape);
			}

			//win.clear_overlay();
			//win.set_image(img);
			//win.add_overlay(render_face_detections(shapes));


			dlib::array<array2d<rgb_pixel> > face_chips;
			dlib::array2d<rgb_pixel> chip;
			extract_image_chips(img, get_face_chip_details(shapes), face_chips);
			extract_image_chip(face_chips[0], rectangle(20, 20, 179, 179), chip);
			//win_faces.set_image(tile_images(face_chips));
			int count = 0;
			for (count; count< face_chips.size(); ++count)
			{
				/*stringstream ss;
				ss << count;
				string str = argv[2];*/
				save_jpeg(chip, dname);
			}

			//cout << "Hit enter to process the next image..." << endl;
			//cin.get();
		}
	}
	catch (exception& e)
	{
		cout << "\nexception thrown!" << endl;
		cout << e.what() << endl;
	}
}


