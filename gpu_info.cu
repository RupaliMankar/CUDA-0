#include <iostream>
#include <fstream>
#include <string>

//int fwrite(std::string, prop);


int main(int argc, char* argv[]){

	cudaError_t error;
	cudaDeviceProp prop;
	int count;													//stores the number of CUDA compatible devices
	std::string filename;

//see if user has given any file name otherwise give default file name
	if(argc==2){
		 filename = argv[1];
		std::cout<<"Output file name is "<<argv[1]<<std::endl;
		
	}else{
		std::cout<<"Default file name is 'gpu_properties.txt'"<<std::endl;
		filename = "gpu_properties.txt";
	}
	
	error = cudaGetDeviceCount(&count);							//get the number of devices with compute capability < 1.0

	if(error != cudaSuccess){												//if there is an error getting the device count
		std::cout<<"ERROR calling cudaGetDeviceCount()"<<std::endl;	//display an error message
		return error;														//return the error
	}

	std::cout<<"Number of CUDA devices: "<<count<<std::endl;
	std::cout<<"Device 0 Properties-------------------------"<<std::endl;
	
	error = cudaGetDeviceProperties(&prop, 0);					//get the properties for the first CUDA device

	if(error != cudaSuccess){												//if there is an error getting the device properties
		std::cout<<"ERROR calling cudaGetDeviceProperties()"<<std::endl;	//display an error message
		return error;														//return the error
	}

	std::ofstream outfile (filename);                       //creating output file stream
	
//writing to the file
	outfile<<"Name:                  "<<prop.name<<std::endl
			 <<"Global Memory:         "<<(double)prop.totalGlobalMem/1024/1000000<<" Gb"<<std::endl
			 <<"Shared Memory/block:   "<<(double)prop.sharedMemPerBlock/1024<<" Kb"<<std::endl
			 <<"Registers/block:       "<<prop.regsPerBlock<<std::endl
			 <<"Warp Size:             "<<prop.warpSize<<std::endl
			 <<"Max Threads/block:     "<<prop.maxThreadsPerBlock<<std::endl
			 <<"Max Block Dimensions:  ["
			 						  <<prop.maxThreadsDim[0]<<" x "
			 						  <<prop.maxThreadsDim[1]<<" x "
			 						  <<prop.maxThreadsDim[2]<<"]"<<std::endl
			 <<"Max Grid Dimensions:   ["
			 						  <<prop.maxGridSize[0]<<" x "
			 						  <<prop.maxGridSize[1]<<" x "
			 						  <<prop.maxGridSize[2]<<"]"<<std::endl
			 <<"Constant Memory:       "<<(double)prop.totalConstMem/1024<<" Kb"<<std::endl
			 <<"Compute Capability:    "<<prop.major<<"."<<prop.minor<<std::endl
			 <<"Clock Rate:            "<<(double)prop.clockRate/1000000<<" GHz"<<std::endl;


	outfile.close();     //closing file output file stream

}

//int fwrite(std::string outfile,&prop)
