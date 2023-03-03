# Operationalizing-an-AWS-ML-Project

This project use tools and SageMaker features to adjust, improve, configure, and prepare a image classification model for production-grade deployment.

The right configuration for deployment is a very important step in machine learning operations as its can avoid problems such as high costs and bad performance. Some examples of configurations for production deployment of a model includes computer resources such as machine instance type and number of instances for training and deployment, security since poor security configuration can leads to data leaks or performance issues. By implement the right configuration we can have a high-throughtput and low-lantecy machine learning model in production.

 Train and deploy an image classification model on AWS Sagemaker


Setup notebook instance


Setup S3 

Finding s3 

![finds3](https://user-images.githubusercontent.com/94936606/222781323-66d0ac89-a9d2-4db1-a1fc-b5c0385dccbf.PNG)


As we can see our bucket was created in S3 
![s3bucket](https://user-images.githubusercontent.com/94936606/222781516-406d5a78-8453-4af3-8cc0-fec6b80149df.PNG)


### Uploading data to S3

The snipped code bellow shows how to donwload data using wget command and upload it to AWS s3 using the cp command
 
 ```
 %%capture
!wget https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip
!unzip dogImages.zip
!aws s3 cp dogImages s3://mlopsimageclassification/data/ --recursive
```

Bellow can see that data was successfuly uploaded to s3

![datains3](https://user-images.githubusercontent.com/94936606/222781235-125d4a7f-a07b-4402-b98e-820dbdef8ea7.PNG)


Training model

```
os.environ['SM_CHANNEL_TRAINING']='s3://mlopsimageclassification/data/'
os.environ['SM_MODEL_DIR']='s3://mlopsimageclassification/model/'
os.environ['SM_OUTPUT_DATA_DIR']='s3://mlopsimageclassification/output/'
tuner.fit({"training": "s3://mlopsimageclassification/data/"})
```

We can see the training job status at SageMaker -> Training Jobs

