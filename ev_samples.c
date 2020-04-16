//OpenCL 事件的使用，以及回调函数
//▶ 事件的两种使用方法。第一种是用事件 a 标记进入命令队列的操作 A，
//于是后续进入命令队列的操作 B 可以被要求等到前面事件 a 完成（即操作 A 完成）以后才能开始调度执行
//。第二种是使用用户自定义的事件创造和标记完成操作来手动控制时间，阻塞任务的进行。
//● 事件的使用代码（用两向量之和的代码改过来的）

#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

const int nElement = 2048;

const char *programSource = "                                              \
	__kernel void vectorAdd(__global int *A, __global int *B, __global int *C) \
	 {                                                                          \
	      int idx = get_global_id(0);                                            \
	      C[idx] = A[idx] + B[idx];                                              \
	      return;                                                                \
	  }                                                                          \
	 ";

//ADD CALLBACK FUN
void hostFunction(int data)
{
	printf("<hostFunction> data= %d\n",data);
}

void CL_CALLBACK callbackFunction(cl_event eventIn,cl_int status,void *userData)
{
	hostFunction(*(int*)userData);
}
int  main()
{
	cl_platform_id *listPlatform;
	const size_t datasize = sizeof(int)*nElement;
	int i,*A,*B,*C;
	cl_int status;
	cl_event eventList[3];

	A = (int *)malloc(datasize);
	B = (int *)malloc(datasize);
	C = (int *)malloc(datasize);
	for(i =0; i< nElement;A[i]=B[i]=i,i++);

	cl_uint nPlatform;
	clGetPlatformIDs(0,NULL,&nPlatform);
	listPlatform = (cl_platform_id *)malloc(nPlatform *sizeof(cl_platform_id));
	clGetPlatformIDs(nPlatform,listPlatform,NULL);
	cl_uint nDevice =0;
	clGetDeviceIDs(listPlatform[0],CL_DEVICE_TYPE_ALL,0,NULL,&nDevice);
	cl_device_id *listDevice=(cl_device_id*)malloc(nDevice*sizeof(cl_device_id));
	clGetDeviceIDs(listPlatform[0],CL_DEVICE_TYPE_ALL,nDevice,listDevice,NULL);

	cl_context context = clCreateContext(NULL,nDevice,listDevice,NULL,NULL,&status);
	cl_command_queue cmdQueue=clCreateCommandQueue(context,listDevice[0],0,&status);
	cl_program program=clCreateProgramWithSource(context,1,(const char**)&programSource,NULL,&status);
	status = clBuildProgram(program,nDevice,listDevice,NULL,NULL,NULL);
	cl_kernel kernel = clCreateKernel(program,"vectorAdd",&status);
	cl_mem bufferA,bufferB,bufferC;

	bufferA = clCreateBuffer(context,CL_MEM_READ_ONLY,datasize,NULL,&status);
	bufferB = clCreateBuffer(context,CL_MEM_READ_ONLY,datasize,NULL,&status);
	bufferC = clCreateBuffer(context,CL_MEM_WRITE_ONLY,datasize,NULL,&status);

	eventList[0] = clCreateUserEvent(context,&status); //User defined events
	clEnqueueWriteBuffer(cmdQueue,bufferA,CL_FALSE,0,datasize,A,1,&eventList[0],&eventList[1]);
	clEnqueueWriteBuffer(cmdQueue,bufferB,CL_FALSE,0,datasize,B,1,&eventList[0],&eventList[2]);

	clSetKernelArg(kernel,0,sizeof(cl_mem),&bufferA);
	clSetKernelArg(kernel,1,sizeof(cl_mem),&bufferB);
	clSetKernelArg(kernel,2,sizeof(cl_mem),&bufferC);

	size_t globalSize[1]={nElement},localSize[1]={256};
	clEnqueueNDRangeKernel(cmdQueue,kernel,1,NULL,globalSize,localSize,3,eventList,NULL); //核函数的调用需要等待事件列表，事件列表长度为3
	//clSetEventCallback(eventList[0],CL_COMPLETE,callbackFunction,&i); //在自定义事件 eventList[0] 完成后允许回调函数开始执行
	clSetUserEventStatus(eventList[0],CL_COMPLETE);//自定义完成事件，eventList[0],这样一来写缓冲区和内核才能开始运行
	clSetEventCallback(eventList[0],CL_COMPLETE,callbackFunction,&i); //在自定义事件 eventList[0] 完成后允许回调函数开始执行
	clEnqueueReadBuffer(cmdQueue,bufferC,CL_TRUE,0,datasize,C,0,NULL,NULL);
	for(i =0;i<nElement;i++)
	{
		if(C[i]!=i+i){
			break;
		}
	}

	printf("Out is %s.\n",(i==nElement? "correct":"incorrect"));
	free(A);
	free(B);
	free(C);
	free(listPlatform);
	free(listDevice);
	clReleaseContext(context);
	clReleaseMemObject(bufferA);
	clReleaseMemObject(bufferB);
	clReleaseMemObject(bufferC);
	clReleaseCommandQueue(program);
	clReleaseKernel(kernel);
	getchar();
	return 0;
}
