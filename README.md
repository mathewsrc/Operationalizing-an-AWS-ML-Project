# Operationalizing-an-AWS-ML-Project

This project use tools and SageMaker features to adjust, improve, configure, and prepare a image classification model for production-grade deployment.

The right configuration for deployment is a very important step in machine learning operations as its can avoid problems such as high costs and bad performance. Some examples of configurations for production deployment of a model includes computer resources such as machine instance type and number of instances for training and deployment, security since poor security configuration can leads to data leaks or performance issues. By implement the right configuration we can have a high-throughtput and low-lantecy machine learning model in production.

 Train and deploy an image classification model on AWS Sagemaker


## Setup notebook instance

Finding SageMaker in AWS

![findsagemaker](https://user-images.githubusercontent.com/94936606/222783294-3d42a7cb-32bf-498a-916c-95feefe29724.PNG)


In SageMaker we then create a notebook instance by looking for Notebook -> Notebook Instances

![createanotebookinstance](https://user-images.githubusercontent.com/94936606/222784137-37d422bd-db21-4bb1-89ff-70dca3523689.PNG)


Then we create a new instance choosing a notebook instance name and type

![setupnotebookinstance](https://user-images.githubusercontent.com/94936606/222784354-13b28a70-3078-40a0-93e0-e1e0622d34d1.PNG)


Bellow you can see a notebook instance called mlops already created

![notebookinstance](https://user-images.githubusercontent.com/94936606/222784785-bca7aae8-4c4c-4c6b-b26b-76b9278b511e.PNG)


## Setup S3 

Finding s3 

![finds3](https://user-images.githubusercontent.com/94936606/222781323-66d0ac89-a9d2-4db1-a1fc-b5c0385dccbf.PNG)


Next, we create a new bucket by clicking in create a new bucket button and give our S3 bucket a unique name

![creates3bucket](https://user-images.githubusercontent.com/94936606/222782172-43b17e3c-fa2a-4df6-907a-dfad5605576b.PNG)


As we can see our bucket was created in S3 

![s3bucket](https://user-images.githubusercontent.com/94936606/222781516-406d5a78-8453-4af3-8cc0-fec6b80149df.PNG)


Uploading data to S3

The snipped code bellow shows how to donwload data using wget command and upload it to AWS s3 using the cp command
 
 ```
 %%capture
!wget https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip
!unzip dogImages.zip
!aws s3 cp dogImages s3://mlopsimageclassification/data/ --recursive
```

Bellow can see that data was successfuly uploaded to s3

![datains3](https://user-images.githubusercontent.com/94936606/222781235-125d4a7f-a07b-4402-b98e-820dbdef8ea7.PNG)


## Training model

### Defining enviroment variables for hyperparameter tunning

SM_CHANNEL_TRAINING: where the data used to train model is located in AWS S3
SM_MODEL_DIR: where model artifact will be saved in S3
SM_OUTPUT_DATA_DIR: where output will be saved in S3

```
os.environ['SM_CHANNEL_TRAINING']='s3://mlopsimageclassification/data/'
os.environ['SM_MODEL_DIR']='s3://mlopsimageclassification/model/'
os.environ['SM_OUTPUT_DATA_DIR']='s3://mlopsimageclassification/output/'
tuner.fit({"training": "s3://mlopsimageclassification/data/"})
```

For this model two hyperparameters was tunning: learning rate and batch size.
```
hyperparameter_ranges = {
    "learning_rate": ContinuousParameter(0.001, 0.1),
    "batch_size": CategoricalParameter([32, 64, 128, 256, 512]),
}
```

Bellow you can see how hyperparameter tuner and estimator was defined. Notice that we are using a py script as entry point to the estimator, this script contains the code need to train model with different hyperparameters values.

```
estimator = PyTorch(
    entry_point="hpo.py",
    base_job_name='pytorch_dog_hpo',
    role=role,
    framework_version="1.4.0",
    instance_count=1,
    instance_type="ml.g4dn.xlarge",
    py_version='py3'
)

tuner = HyperparameterTuner(
    estimator,
    objective_metric_name,
    hyperparameter_ranges,
    metric_definitions,
    max_jobs=2,
    max_parallel_jobs=1,  # you once have one ml.g4dn.xlarge instance available
    objective_type=objective_type
)
```


We can see the training job status at SageMaker -> Training Jobs

