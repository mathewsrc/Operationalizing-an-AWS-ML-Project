# Operationalizing-an-AWS-ML-Project

**Project developed for AWS Machine Learning Engineer Scholarship offered by Udacity (2023)**


The right configuration for deployment is a very important step in machine learning operations as its can avoid problems such as high costs and bad performance. Some examples of configurations for production deployment of a model includes computer resources such as machine instance type and number of instances for training and deployment, security since poor security configuration can leads to data leaks or performance issues. By implement the right configuration we can have a high-throughtput and low-lantecy machine learning model in production.

---------

## Setup notebook instance

Finding SageMaker in AWS

![findsagemaker](https://user-images.githubusercontent.com/94936606/222783294-3d42a7cb-32bf-498a-916c-95feefe29724.PNG)


In SageMaker we then create a notebook instance in Notebook -> Notebook Instances -> Create notebook instance button

![createanotebookinstance](https://user-images.githubusercontent.com/94936606/222784137-37d422bd-db21-4bb1-89ff-70dca3523689.PNG)


Then we create a new instance choosing a notebook instance name and type. For this project ml.m5.xlarge instance type was selected

![setupnotebooksetinstance](https://user-images.githubusercontent.com/94936606/222964999-57346d9c-4243-40e8-891a-c53e359436dc.png)


Bellow you can see a notebook instance called mlops already created

![notebookcreated](https://user-images.githubusercontent.com/94936606/222965050-f08bcb83-7111-4d16-991f-a5a5401e313f.png)

With the notebook instance create we can upload a jupyter notebook, the script.py file to train our deep learning image classification model, the inference.py script and an image for test. Bellow you can find the list of files that we need to upload:

train_and_deploy-solution.ipynb
hpo.py
inference2.py
lab.jpg

Before we can start to train our model we need first create a S3 bucket where we will upload our train, validation and test data to. So let's do it now :)


---------

## Setup S3 

Finding s3 

![finds3](https://user-images.githubusercontent.com/94936606/222781323-66d0ac89-a9d2-4db1-a1fc-b5c0385dccbf.PNG)


Next, we create a new bucket by clicking in create a new bucket button and give our S3 bucket a unique name

![creates3bucket](https://user-images.githubusercontent.com/94936606/222782172-43b17e3c-fa2a-4df6-907a-dfad5605576b.PNG)


As we can see our bucket was created in S3 

![s3bucket](https://user-images.githubusercontent.com/94936606/222781516-406d5a78-8453-4af3-8cc0-fec6b80149df.PNG)


### Uploading data to S3

The snipped code bellow shows how to donwload data using wget command and upload it to AWS s3 using the cp command.
 
 ```bash
 %%capture
!wget https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip
!unzip dogImages.zip
!aws s3 cp dogImages s3://mlopsimageclassification/data/ --recursive
```
**Note:  This code are located in the train_and_deploy.ipynb**

Bellow can see that data was successfuly uploaded to s3

![datains3](https://user-images.githubusercontent.com/94936606/222781235-125d4a7f-a07b-4402-b98e-820dbdef8ea7.PNG)

Now that we have our data in S3 we can train our model. 

So let's starting by reviewing some important information you will see in the jupyter notebook

---------

## Hyperparameter tunning

### Defining enviroment variables for hyperparameter tunning

SM_CHANNEL_TRAINING: where the data used to train model is located in AWS S3

SM_MODEL_DIR: where model artifact will be saved in S3

SM_OUTPUT_DATA_DIR: where output will be saved in S3


Here we are passing some paths to our S3 which will be used by the notebook instance to get data, save model and output
```python
os.environ['SM_CHANNEL_TRAINING']='s3://mlopsimageclassification/data/'
os.environ['SM_MODEL_DIR']='s3://mlopsimageclassification/model/'
os.environ['SM_OUTPUT_DATA_DIR']='s3://mlopsimageclassification/output/'
```

Here we can see how we can access the enviroment variables in hpo.py script

```python
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--data', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    
    args=parser.parse_args()
```

For this model two hyperparameters was tunning: learning rate and batch size.

```python
hyperparameter_ranges = {
    "learning_rate": ContinuousParameter(0.001, 0.1),
    "batch_size": CategoricalParameter([32, 64, 128, 256, 512]),
}
```

Bellow you can see how hyperparameter tuner and estimator was defined. Notice that we are using a py script (hpo.py) as entry point to the estimator, this script contains the code need to train model with different hyperparameters values.

```python
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

tuner.fit({"training": "s3://mlopsimageclassification/data/"})
```

**Note: we are passing a S3 path where the data for training, validation and testing are loacated to the HyperparameterTuner fit method**



After we start the model training we can see the training job status at SageMaker -> Training -> Training Jobs

![trainingjobs](https://user-images.githubusercontent.com/94936606/222787406-59f41c0f-57ea-4227-ab7d-3caecdbbda8b.PNG)


## Training Model with best hyperparameters values

**Without multi-instance**

Notice that training a model without enable multi-instance took 21 minutes to complete


![trainingwithoutmultiinstance](https://user-images.githubusercontent.com/94936606/222796302-0d04321f-567e-4847-a8a9-d994ee4be04c.PNG)

![trainingjobwithoutmultiinstanceconfigs](https://user-images.githubusercontent.com/94936606/222804951-c9385220-28da-4cc4-b6d8-b71f52a0f0ff.PNG)


**Deploying model**

We can check the deployed model in SageMaker -> Inference -> Endpoints

Notice that the model was deployed with one initial instance and a instance type which uses the type ml.m5.large 

```python
predictor = pytorch_model.deploy(initial_instance_count=1, instance_type='ml.m5.large')
```

Bellow we can see the deployed model


![modeldeployedwithoutmultiinstance](https://user-images.githubusercontent.com/94936606/222797949-231ec90b-5dcd-46e4-80a8-8ac8047c85a9.PNG)


**With multi-instance**


![trainingjobmultiinstance](https://user-images.githubusercontent.com/94936606/222804798-5e00d917-8278-46de-8df3-8ccf740e9c4e.PNG)


![trainingjobmultiinstanceconfigs](https://user-images.githubusercontent.com/94936606/222804816-afa72eee-40c7-4037-aca5-803b353b0300.PNG)

---------

## EC2 Setup

EC2 as others AWS services can be founded by search it by name in AWS 

![findec2](https://user-images.githubusercontent.com/94936606/222843668-9fc36a82-23db-4569-a758-7512bd59d59a.PNG)

Now we can create our new instance by clicking in Launch instances button

![ec2instance](https://user-images.githubusercontent.com/94936606/222844655-55c3bb6b-a01b-4f8f-a2e6-b1de681a11a6.PNG)

First, we must give a name to our instance 

![setupec2name](https://user-images.githubusercontent.com/94936606/222932320-875bae01-4a09-4ed5-870f-0cc8975e9953.PNG)


We are now selecting an Amazon Machine Image (AMI), which is a supported and maintained image provided by AWS that contains the necessary information to launch an instance. Since we will be training a deep learning model with PyTorch, we need to select an AMI that supports PyTorch for deep learning.

![setupec2choosingami](https://user-images.githubusercontent.com/94936606/222932853-53268971-ce6e-4a5e-b4fd-0c5aef409d4e.PNG)

We can have an overview of the AMI information in this image

![amidetails](https://user-images.githubusercontent.com/94936606/222844621-a57b98fc-aa0f-4d68-84ed-33896845f477.PNG)


Next, we need to choose an EC2 instance that is supported by this AMI. According to the documentation, this type of AMI supports the following instances: G3, P3, P3dn, P4d, G5, and G4dn. 

![setupec2choosinginstance](https://user-images.githubusercontent.com/94936606/222932534-f18476a7-78ca-49b1-a7b1-050be429bf21.PNG)


EC2 requires a key pair that can be used, for example, to SSH into our instance from another service. A good example would be SSHing into our instance from AWS Cloud9.

![setupec2creatingapairkey](https://user-images.githubusercontent.com/94936606/222932612-0495725f-212a-4ae9-817a-0ae627322cd5.PNG)

![setupec2createkeypair](https://user-images.githubusercontent.com/94936606/222932682-b54fde71-f88d-48cc-bd8e-c2eac453dfac.png)


**Note: to simplify things, other configurations will be set to their default values.**


Now that we created our instance we can connecting to it following the three images bellow


![connectingtoec2choosinginstance](https://user-images.githubusercontent.com/94936606/222932960-31e1d5a2-59ba-4819-9268-b1ec05449c12.PNG)

![ec2connecting](https://user-images.githubusercontent.com/94936606/222932962-516c012f-64ab-4c62-a9a4-684d6fda557e.PNG)

![ec2connecting2](https://user-images.githubusercontent.com/94936606/222932972-34559c9b-d2c0-419e-979a-69257ce63404.PNG)


If everything works well we are connected to our instance. The last step is activate pytorch virtual enviroment by typing source activate pytorch on terminal

![ec2activatepytorchvirtualenviroment](https://user-images.githubusercontent.com/94936606/222933025-818a2860-72ed-4a1e-8af4-c9c3eca08966.PNG)


Now the fun part :)

First we need to donwload the dataset to EC2 by running the following command on terminal

```bash
wget https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zipunzip dogImages.zip
```

Since we are downloading our data to EC2, we can retrieve the path in the training script as follows. This is a key difference between training the model in a notebook instance versus training it on EC2

```python
data = 'dogImages'
train_data_path = os.path.join(data, 'train')
test_data_path = os.path.join(data, 'test')
validation_data_path=os.path.join(data, 'valid')
```

Next, we are going to create a directory to save the trained models

```bash
mkdir TrainedModels
```

Now, we need to create a Python file and paste the training code into it

Use vim to create an empy file

```bash
vim solution.py
```

Use the following command so that we paste our code into solution.py

```bash
:set paste
```

Copy the code located in https://github.com/mathewsrc/Operationalizing-an-AWS-ML-Project/blob/master/ec2train1.py and paste into solution.py

```bash
:wq! + Press Enter
```

Now we can run our script to train the model

```python
python solution.py
```

### EC2 vs Notebook instance for training models

Both services have their own advantages:

EC2 instances can be easily scaled up or down based on computing needs, can be customized to meet specific requirements such as framework (pytroch or tensorflow), number of CPUs, memory size and GPU support and EC2 instances can be optimized for high-performance computing, which can greatly reduce the time it takes to train large machine learning models.

Notebook instances have their own advantages too such as: quick setup as they comes with pre-configured with popular machine learning frameworks and libraries, easy collaboration and integration with others AWS services such as AWS SageMaker, which provides a lot of tools required for machine learning engineering and operations.

------------

## Lambda Functions Setup

The following images show how to create a AWS Lambda Function: 

Finding Lambda Functions

![findlambda](https://user-images.githubusercontent.com/94936606/222853992-631e6366-8885-4dd8-b0d9-64a6ab14411c.PNG)


Creating a Lambda Function

To create a Lambda Function click on Create a function button

![create a function](https://user-images.githubusercontent.com/94936606/222854008-addfa523-2e5d-41d7-bdd8-3c84e1ef0e69.PNG)

Deploying a Lambda Function

To update our Lambda Function we need to click on Deploy button. The Deploy button is located to the right of Test button

![lambdadeployfunction](https://user-images.githubusercontent.com/94936606/222854041-8c45084e-ec59-406d-917b-41f15b88f074.PNG)


Lambda Function configuration

Notice that we have the ability to adjust the memory and storage requirements based on our specific needs.

![lambdaconfiguration](https://user-images.githubusercontent.com/94936606/222854048-4e323084-fc5d-4838-8fd9-e5256433343e.PNG)

After we create the Lambda Function we can replace the default code with the code located in https://github.com/mathewsrc/Operationalizing-an-AWS-ML-Project/blob/master/lamdafunction.py

Since a Lambda function will invoke a SageMaker endpoint, we need to grant permission to the Lambda function to access SageMaker

```python
runtime=boto3.Session().client('sagemaker-runtime')
    
    response=runtime.invoke_endpoint(EndpointName=endpoint_Name,
                                    ContentType="application/json",
                                    Accept='application/json',
                                    #Body=bytearray(x)
                                    Body=json.dumps(bs))
    
    result=response['Body'].read().decode('utf-8')
    sss=json.loads(result)
```


Adding SageMaker access permission to Lambda Function

We need to add a new policy to our Lambda function so that it can access SageMaker. This can be done through AWS IAM.

![findiam](https://user-images.githubusercontent.com/94936606/222854122-f0772d0a-7dee-4810-8e8f-5d300fdca799.PNG)

First select roles

![iamroletab](https://user-images.githubusercontent.com/94936606/222854145-4383cd67-d1e6-4d85-b772-0aee1807a48c.PNG)

Next, we need find our Lambda Function and click on it

![imaselectinglambdafunctionrole](https://user-images.githubusercontent.com/94936606/222934907-7322b2f6-08ce-483c-b3cb-c922324f3369.png)

Click on Add permissions button and then on Attach policies button

![iamaddpermissions](https://user-images.githubusercontent.com/94936606/222854165-88d3f1c9-4277-4dce-9e9d-3f08f09be906.PNG)

Finally, we should search for SageMaker and select an appropriate policy. While the full access option may be the simplest choice, it's important to remember that granting excessive permissions to a service can pose security risks. Therefore, it's advisable to carefully consider the level of access required for your specific use case.

![iamsagemakerpermissionsforlambda](https://user-images.githubusercontent.com/94936606/222854174-188a9f7a-3895-4387-b393-9688e53fd18f.PNG)


Now with the right permission we can create a new test to test our Lambda Function. 

First click on Test button

![testlambdafunction](https://user-images.githubusercontent.com/94936606/222854866-e2337eea-b024-4e7e-8f96-bf0c0b81f6d1.PNG)

Now give a name for the test

![lambdafunctionconfiguringtest](https://user-images.githubusercontent.com/94936606/222935202-2df64dfd-951d-4f7e-bd5a-42ffdfc30ecf.png)

Replace the default JSON with the following JSON data, as shown in the image below

```
{ "url": "https://s3.amazonaws.com/cdn-origin-etr.akc.org/wp-content/uploads/2017/11/20113314/Carolina-Dog-standing-outdoors.jpg" }
```

![addjsontotestlambda](https://user-images.githubusercontent.com/94936606/222935233-27d6b70c-e355-4424-86e0-251d14735996.PNG)


## Scaling Lambda Function and Endpoint

### Adding concurrency to Lambda Function

By default a Lambda Function can only respond one request at once. One way to change that is to use concurrency so that the Lambda Function can be responds to multiple requests at once. Before add concurrency we need to configure a version which can be done in Configuration tab.

![lambdaversionsetup](https://user-images.githubusercontent.com/94936606/222857369-103c2e2a-3d2b-4cc5-98f8-6751a3e19704.PNG)

Add a description for the new version and click on Publish button

![creatinganewversionforlambda](https://user-images.githubusercontent.com/94936606/222935573-323457f8-3146-4ca4-9602-d93c29bbf0fe.png)


Now we can add concurrency in Configuration -> Provisioned concurrency -> Edit button

![provisionedconcurrencyforlambda](https://user-images.githubusercontent.com/94936606/222935601-98a902ec-1ede-4f6d-9cf5-0b91c5c6fee3.png)

Our final task is to select an integer value for concurrency. Provisioned concurrency initializes a specified number of execution environments, enabling them to respond immediately. Therefore, a higher concurrency level results in reduced latency. In this example the concurrency was set to two so this Lambda Functions can handle two requests at once which could not be enough for services with high demand.

![lambdaconcurrencyconfig](https://user-images.githubusercontent.com/94936606/222857390-3522993a-84d0-4aed-a2da-12e7832bd2af.PNG)

---------
### Auto-Scaling endpoint

Auto scaling is a powerful feature of SageMaker that allows for dynamic adjustment of the number of instances used with deployed models based on changes in workload. With auto scaling, SageMaker automatically increases or decreases the number of instances, ensuring that we only pay for the instances that are actively running.

We can enable auto-scaling in SageMaker -> Endpoints -> Endpoint runtime settings

![endpointruntimesettings](https://user-images.githubusercontent.com/94936606/222857499-e5d3a8c5-1d8e-4086-a33e-3e8972fac5bf.PNG)

We can increase the maximum of instances for our endpoint

![autoscalingnumberinstances](https://user-images.githubusercontent.com/94936606/222857543-2ae15526-6ca3-4b5c-b2c0-8646cc32fb7d.PNG)

We can define a scaling policy to control how auto-scaling works. In this example, the 'Target value' was set to 20, meaning that when our endpoint receives 20 requests simultaneously, auto-scaling will be triggered, and the number of instances will be increased. The 'Scale In' and 'Scale Out' parameters were both set to 30 seconds, which controls the amount of time auto-scaling should wait before increasing or decreasing the number of instances.

![scallingpolicy](https://user-images.githubusercontent.com/94936606/222857563-80cec446-1a80-4174-b589-db8579d4cc3f.PNG)

Now we see that our endpoint has auto-scaling enabled

![autoscalingcreated](https://user-images.githubusercontent.com/94936606/222857530-9cde4099-48aa-403e-a093-8bc7b01a7dbd.PNG)

---------

## Deleting EC2 instances, Lambda Functions, and Endpoints

To avoid any cost we can delete all services and instances used in this project. Bellow you can see how to terminate, delete or stop services and instances in AWS.


Deleting the notebook instance. Notebook instances are located in SageMaker -> Notebooks -> Notebook Instances

![stopndeletenotebookinstance](https://user-images.githubusercontent.com/94936606/222857619-da429195-fc35-4b88-be6a-2ebbaea1c268.PNG)


Deleting EC2 instances. EC2 instances are located in EC2 -> Instances

![stopnterminateec2](https://user-images.githubusercontent.com/94936606/222857625-a18d30bb-ce74-4e38-a221-4071e7d099f4.PNG)


Deleting Lambda Functions. Lambda Functions are located in Lambda -> Functions

![deletelambda](https://user-images.githubusercontent.com/94936606/222857648-bb2bbd3f-261c-4135-bab6-c61b59acebc6.PNG)


Deleting Endpoints. Endpoints are located in SageMaker -> Endpoints

![deletingendpoint](https://user-images.githubusercontent.com/94936606/222857661-274b5459-b851-4c6d-acd2-ca62e838fe41.PNG)


## References

https://docs.aws.amazon.com/sagemaker/latest/dg/whatis.html

https://docs.aws.amazon.com/sagemaker/latest/dg/nbi.html

https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/concepts.html

https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AMIs.html

https://docs.aws.amazon.com/s3/index.html?nc2=h_ql_doc_s3

https://docs.aws.amazon.com/lambda/latest/dg/welcome.html

https://docs.aws.amazon.com/lambda/latest/dg/configuration-concurrency.html

https://docs.aws.amazon.com/IAM/latest/UserGuide/introduction.html

https://docs.aws.amazon.com/sagemaker/latest/dg/endpoint-auto-scaling.html
