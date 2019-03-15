#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>



extern "C"
{
#include <mvnc.h>
}

using namespace std;
using namespace cv;

#define SSD_GRAPH_DIR "./ssd.graph"

vector<float> object_info(7,0);   // face_info
const float min_score_percent = 0.2;  // the minimal score for a box to be shown

vector<string> LABELS = {"background", "aeroplane", "vehicle", "bird", "boat",
"bottle", "vehicle", "vehicle", "cat", "chair", "cow", "diningtable",
"dog", "horse", "vehicle", "person", "pottedplant", "sheep",
"sofa", "train", "tvmonitor"};


// network image resolution
const int NETWORK_IMAGE_WIDTH = 300;
const int NETWORK_IMAGE_HEIGHT = 300;


// enable networks
bool enableSSDNetwork = true;

// device setup and preprocessing variables
double networkMean[3] = {127.5,127.5,127.5};
double networkStd[3] = {0.007843,0.007843,0.007843};
const uint32_t MAX_NCS_CONNECTED = 1;
uint32_t numNCSConnected = 0;
ncStatus_t retCode;
struct ncDeviceHandle_t* dev_handle[MAX_NCS_CONNECTED];
struct ncGraphHandle_t* sphere_graph_handle = NULL;
struct ncGraphHandle_t* ssd_graph_handle = NULL;
struct ncFifoHandle_t* sphere_fifo_in = NULL;
struct ncFifoHandle_t* sphere_fifo_out = NULL;
struct ncFifoHandle_t* ssd_fifo_in = NULL;
struct ncFifoHandle_t* ssd_fifo_out = NULL;


/**
    overlays the boxes and labels onto the display image
    @param: image
    @param: object_info: [image_id, class_id, score, x1, y1, x2, y2] 
*/
void overlay_on_image( const Mat& image, const vector<float>& object_info) 
{
    Mat display_image = image;
    int source_image_width = display_image.size[1];
    int source_image_height = display_image.size[0];

    int base_index = 0;
    int image_id = object_info[base_index + 0];
    int class_id = object_info[base_index + 1];
    float percentage = int(object_info[base_index + 2] * 100);
    if ((percentage > min_score_percent)&&((class_id == 15)||(class_id == 2)||(class_id == 4)||(class_id == 6)||(class_id == 7)||(class_id == 14))) // ignore boxes less than the minimum score
    {
        stringstream ss;
        ss << LABELS[class_id];
        ss << percentage;
        ss << "%";
        string label_text = ss.str();
        float box_left = int(object_info[base_index + 3] * source_image_width);
        float box_top = int(object_info[base_index + 4] * source_image_height);
        float box_right = int(object_info[base_index + 5] * source_image_width);
        float box_bottom = int(object_info[base_index + 6] * source_image_height);

        int box_thickness = 2;
        rectangle(display_image, Point(box_left, box_top), Point(box_right, box_bottom), Scalar(0,0,255), box_thickness);
        // cout << image_id <<" "<<" "<<class_id<<" "<<percentage<<" "<<box_left<<" "<<box_top<<" "<<" "<<box_right<<" "<<box_bottom<< endl;
    
        // draw the classification label string just above and to the left of the rectangle
		// label background
		Size label_size = getTextSize(label_text, FONT_HERSHEY_SIMPLEX, 0.5, 1, nullptr);  //[0]
        float label_left = box_left;
        float label_top = box_top - label_size.height;
        if (label_top < 1)
	       {label_top = 1;}
        float label_right = label_left + label_size.width;
        float label_bottom = label_top + label_size.height;
        rectangle(display_image, Point(label_left - 1, label_top - 1), Point(label_right + 1, label_bottom + 1), Scalar(125,175,75), -1);

        // label text above the box
        putText(display_image, label_text, Point(label_left, label_bottom), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255,255,255), 1);        
    }
}


/**
 * @brief read_graph_from_file
 * @param graph_filename [IN} is the full path (or relative) to the graph file to read.
 * @param length [OUT] upon successful return will contain the number of bytes read
 *        which will correspond to the number of bytes in the buffer (graph_buf) allocated
 *        within this function.
 * @param graph_buf [OUT] should be set to the address of a void pointer prior to calling
 *        this function.  upon successful return the void* pointed to will point to a
 *        memory buffer which contains the graph file that was read from disk.  This buffer
 *        must be freed when the caller is done with it via the free() system call.
 * @return true if worked and program should continue or false there was an error.
 */
bool read_graph_from_file(const char *graph_filename, unsigned int *length_read, void **graph_buf)
{
    FILE *graph_file_ptr;

    *graph_buf = nullptr;

    graph_file_ptr = fopen(graph_filename, "rb");
    if (graph_file_ptr == nullptr) {
        return false;
    }

    // get number of bytes in file
    *length_read = 0;
    fseek(graph_file_ptr, 0, SEEK_END);
    *length_read = ftell(graph_file_ptr);
    rewind(graph_file_ptr);

    if(!(*graph_buf = malloc(*length_read))) {
        // couldn't allocate buffer
        fclose(graph_file_ptr);
        return false;
    }

    size_t to_read = *length_read;
    size_t read_count = fread(*graph_buf, 1, to_read, graph_file_ptr);

    if(read_count != *length_read) {
        // didn't read the expected number of bytes
        fclose(graph_file_ptr);
        free(*graph_buf);
        *graph_buf = nullptr;
        return false;
    }
    fclose(graph_file_ptr);

    return true;
}

void initNCS(){
    for (int i = 0; i < MAX_NCS_CONNECTED; i++) {
        //initialize device handles
        struct ncDeviceHandle_t* dev;
        dev_handle[i] == dev;
        retCode = ncDeviceCreate(i, &dev_handle[i]);
        if (retCode != NC_OK) {
            if (i == 0) {
                cout << "Error - No neural compute device found." << endl;
            }
            break;
        }

        //open device
        retCode = ncDeviceOpen(dev_handle[i]);
        if (retCode != NC_OK) {
            cout << "Error[" << retCode << "] - could not open device at index " << i << "." << endl;
        }
        else {
            numNCSConnected++;
        }

    }

    if (numNCSConnected > 0) {
        cout << "Num of neural compute devices connected: " << numNCSConnected << endl;
    }
}

void initSsdNetwork() {
    // Setup for Gender network
    if (enableSSDNetwork) {
        
        // read the gender graph from file:
        unsigned int graph_len = 0;
        void *gender_graph_buf;
        if (!read_graph_from_file(SSD_GRAPH_DIR, &graph_len, &gender_graph_buf)) {
            // error reading graph
            cout << "Error - Could not read graph file from disk: " << endl;
            exit(1);
        }

        // initialize the graph handle
        retCode = ncGraphCreate("ssdGraph", &ssd_graph_handle);

        // allocate the graph data type fp32
        retCode = ncGraphAllocateWithFifosEx(dev_handle[0], ssd_graph_handle, gender_graph_buf, graph_len,
                                           &ssd_fifo_in, NC_FIFO_HOST_WO, 2, NC_FIFO_FP32,
                                           &ssd_fifo_out, NC_FIFO_HOST_RO, 2, NC_FIFO_FP32);

	//retCode = ncGraphAllocateWithFifos(dev_handle[0], ssd_graph_handle, gender_graph_buf, graph_len, &ssd_fifo_in,&ssd_fifo_out);
        if (retCode != NC_OK) {
            cout << "Error[" << retCode << "]- could not allocate ssd network." << endl;
            exit(1);
        }
        else {
            cout << "Successfully allocated ssd graph to device 0." << endl;
        }

    }
}

bool getSSDResults(const Mat& inputMat, struct ncGraphHandle_t* graphHandle, struct ncFifoHandle_t* fifoIn,
                                   struct ncFifoHandle_t* fifoOut) {
    cv::Mat preprocessed_image_mat;
    resize(inputMat, preprocessed_image_mat,Size(NETWORK_IMAGE_WIDTH,NETWORK_IMAGE_HEIGHT), CV_INTER_AREA);
    if (preprocessed_image_mat.rows != NETWORK_IMAGE_HEIGHT ||
        preprocessed_image_mat.cols != NETWORK_IMAGE_WIDTH) {
        cout << "Error - preprocessed image is unexpected size!" << endl;
        return -1;
    }

    // three values for each pixel in the image. one value for each color channel RGB
    float tensor32[3];
    float tensor16[NETWORK_IMAGE_WIDTH * NETWORK_IMAGE_HEIGHT * 3];

    uint8_t* image_data_ptr = (uint8_t*)preprocessed_image_mat.data;
    int chan = preprocessed_image_mat.channels();


    int tensor_index = 0;
    for (int row = 0; row < preprocessed_image_mat.rows; row++) {
        for (int col = 0; col < preprocessed_image_mat.cols; col++) {

            int pixel_start_index = row * (preprocessed_image_mat.cols + 0) * chan + col * chan; // TODO: don't hard code

            // assuming the image is in BGR format here
            uint8_t blue = image_data_ptr[pixel_start_index + 0];
            uint8_t green = image_data_ptr[pixel_start_index + 1];
            uint8_t red = image_data_ptr[pixel_start_index + 2];

            //image_data_ptr[pixel_start_index + 2] = 254;

            // then assuming the network needs the data in BGR here.
            // also subtract the mean and multiply by the standard deviation from stat.txt file
            tensor32[0] = (((float_t)blue - networkMean[0]) * networkStd[0]);
            tensor32[1] = (((float_t)green - networkMean[1]) * networkStd[1]);
            tensor32[2] = (((float_t)red - networkMean[2]) * networkStd[2]);


			tensor16[tensor_index++] =  tensor32[0];
			tensor16[tensor_index++] =  tensor32[1];
			tensor16[tensor_index++] =  tensor32[2];
        }
    }

    // queue for inference
    unsigned int inputTensorLength = NETWORK_IMAGE_HEIGHT * NETWORK_IMAGE_WIDTH * 3 * sizeof(float);
    retCode = ncGraphQueueInferenceWithFifoElem(graphHandle, fifoIn, fifoOut, tensor16,  &inputTensorLength, 0);
    if (retCode != NC_OK) {
        cout << "Error[" << retCode << "] - could not queue inference." << endl;
     
        return -1;
    }

    // get the size of the result
    unsigned int res_length;
    unsigned int option_length = sizeof(res_length);
    retCode = ncFifoGetOption(fifoOut, NC_RO_FIFO_ELEMENT_DATA_SIZE, &res_length, &option_length);
    if (retCode != NC_OK) {
        cout << "Error[" << retCode << "] - could not get output result size." << endl;
        return -1;
    }

    float result_buf[res_length];
    void* user_data;
    retCode = ncFifoReadElem(fifoOut, result_buf, &res_length, &user_data);
    if (retCode != NC_OK) {
        cout << "Error[" << retCode << "] - could not get output result." << endl;
        return -1;
    }


    res_length /= sizeof(unsigned short);

    /* make this array large enough to hold the larger result of age/gender network results */
    float result_fp32[res_length];

   // fp16tofloat(result_fp32, (unsigned char*)result_buf, res_length);

    int num_valid_boxes = result_buf[0];

    for(int i=0;i<num_valid_boxes;i++)
    {
        int base_index = 7 + i*7;
        object_info[0] = result_buf[base_index+0];
        object_info[1] = result_buf[base_index+1];
        object_info[2] = result_buf[base_index+2];
        object_info[3] = result_buf[base_index+3];
        object_info[4] = result_buf[base_index+4];
        object_info[5] = result_buf[base_index+5];
        object_info[6] = result_buf[base_index+6];


        overlay_on_image(inputMat,object_info);
    }
    return 1;
}


int main(int argc, char **argv)
{

    initNCS();
    initSsdNetwork();



	cv::Mat image;
	cv::VideoCapture cap(0);
	if (!cap.isOpened())
	{

		cout << "fail to open" << endl;
		return 0;
	}


	while (true)
	{

		if (!cap.read(image))
		{
			cout << "Video is end!" << endl;
			break;
		}
		if (cv::waitKey(1) == 27)
		{
			break;
		}
		//double t1 = (double)cv::getTickCount();
        getSSDResults(image,ssd_graph_handle, ssd_fifo_in, ssd_fifo_out);
		//double t2 = (double)cv::getTickCount();
		//cout<<(double)(t2 - t1) / cv::getTickFrequency()<<"s"<<endl;


	
		cv::imshow("image", image);
		cv::waitKey(1);

	}

	return 0;
}



