# Covid19 Forecasting
### Lang Qin (QA: Yuwei Deng)

## Content
- [Vision](#vision)
- [Mission](#mission)
- [Criteria](#criteria)
- [Planning](#planning)
- [Instructions](#instructions)


## <a name="#vision"></a>Vision

COVID-19 gets more serious and people around the world are paying more attention to the problem. As data scientists, we wish to contribute to this problem by looking for better methods for estimates that can assist medical and governmental institutions to prepare and adjust as pandemics unfold. Data could probably provide valuable insights to the transmission rate. This project is inspired by a [kaggle competition](https://www.kaggle.com/c/covid19-global-forecasting-week-3/overview/evaluation), which aims to answer the questions of 'What is the transmission rate in one week ahead, and 'What are the important factors that affect the transmission rate'. The result will be presented by a web application, which predicts the COVID-19 transmission rate in the future given the historical data.

## <a name="#mission"></a>Mission

This project is designed to forecast confirmed cases and fatalities in different countries, based on the current number of cases and various development indicators of the country. The data of confirmed cases and fatality rate comes from [John Hopkins CSSE](https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_time_series). The data of development indicator comes from [World Bank](http://wdi.worldbank.org/table/2.12#). Both datasets are open-sourced and could be used for research purpose.

The project has two initiatives, model development and web application development. For the first initiative, machine-learning based forecasting model is expected to be built. Moreover, the feature importance is expected to be analyzed to identify factors that appear to impact the transmission rate of COVID-19. For the second initiative, a web application of a sandbox will be developed, which allows its users to enter historical data and various indicators, to make a prediction of the confirmed cases and fatality rates for the next week.

## <a name="#criteria"></a> Success criteria

The machine learning performance metric is [Root mean squared logarithmic error](https://www.kaggle.com/wiki/RootMeanSquaredLogarithmicError), which will be used to assess the model's performace. The final score is the mean of the RMSLE over fatalies and confirmed cases for all countries. The lower the RMSLE, the better the result. The desirable result of RMSLE equal or smaller than 0.2, meaning the square root of log error of the prediction is not more than 20%.

The metric that would actually measure the business outcome is transmission rate. If proper factors are identified by the model and addressed by the government, there could be a slower increase or even reduction in the transmission rate. However, the nature of this project determines its business outcome is difficult to measure.


## <a name="#planning">Planning

### Initiative 1：Forecast Model Development
- **Epic 1**: Problem Exploration
	-   **Story 1**: Literature review 
		- Reading on how others model the virus transmission
		- Identify potential risks associated with the project
    -   **Story 2**: Looking for supplementary datasets
    
-   **Epic 2**:  Data Preprocessing
	- **Story 1**: Data Exploratory Analysis
		- Looking at the distribution of the data
		- Looking at the correlation of features
	- **Story 2**: Cleaning the data
		- Dealing with the missing values
		- Dealing with extreme values
		- Dealing with categorical features
	- **Story 3**: Deriving additional features
	- **Story 4**: Splitting for training, testing and validation dataset

-   **Epic 3**:  Model Development
	- **Story 1**: Building a baseline model
		- Build a regression model
	- **Story 2**: Developing evaluation metrics
	- **Story 3**: Backtesting the model
	- **Story 4**: Building more models
	- **Story 5**: Comparing the models
	-  **Story 6**: Operationalizing models
		- Obtaining weekly-updated dataset
		- Retraining the models every week

-   **Epic 4**:  Feature Analysis
	- **Story 1**: Analyzing the feature importance
		- Sensitivity tests of each feature
		- Ranking the importance

### Initiative 2: Web Application Development

-  **Epic 1**:  Design the Application Structure
-  **Epic 2**: Front-end Development
	- **Story 1**: Designing the  Framework
	- **Story 2**: Coding the front-end
- **Epic 3**: Back-end Development
	-  **Story 1**: Setting up RDS to query from the app
	-  **Story 2**: Setting up S3 to store raw data
	-  **Story 3**: Online deployment on AWS
- **Epic 4**: QA Testing


### Backlog

1.  “Initiative1.epic1.story1” (2 points) - PLANNED
2.  “Initiative1.epic1.story2” (1 points) - PLANNED
3.  “Initiative1.epic2.story1” (2  points) - PLANNED
4. “Initiative1.epic2.story2” (4 points) - PLANNED
5. “Initiative1.epic2.story3” (4 points) - PLANNED
6. “Initiative1.epic2.story4” (2 points) - PLANNED
7. “Initiative1.epic3.story1” (2 points)
8. “Initiative1.epic3.story2” (1 points)
9. “Initiative1.epic3.story3” (1 points) 
10. “Initiative1.epic3.story4” (8 points)
11. “Initiative1.epic3.story5” (4 points) 
12. “Initiative1.epic3.story6” (8 points)  
13. “Initiative1.epic4.story1” (8 points)  
14. “Initiative2.epic1” (2 points)  
15. “Initiative2.epic2.story1” (1 points)  
16. “Initiative2.epic2.story2” (8 points)  
17. “Initiative2.epic3.story1” (8 points)  

### Icebox
1. “Initiative2.epic3.story2” 
2. “Initiative2.epic3.story3” 
3. “Initiative2.epic4” 

## <a name="#instructions">Instructions

### 1. Build the docker from terminal.
```
cd 2020-msia423-Qin-Covid19
docker build -t runapp .
```
### 2. Download the data from [John Hopkins CSSE]([https://github.com/CSSEGISandData/COVID-19/blob/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv]) (Static, public data source).
```
docker run --mount type=bind,source="$(pwd)"/data,target=/app/data runapp python3 src/download_data.py
```
By default, the data will be downloaded to data/sample, you can also specify the output path by changing the OUTPUT_PATH in src/config.py.

### 3. Upload the data to AWS S3 bucket.

To connect to the AWS S3, you need to change the credential information in config/aws_s3.conf by replacing \<aws access key> and \<aws secret access key> with your own key, which can be found in "security_credentials" section under your AWS account.
```
docker run --mount type=bind,source="$(pwd)"/data,target=/app/data runapp python3 src/upload_data.py
```
By default the bucket is "nw-langqin-s3", you can also specify your own own bucket by add a ```--bucket``` or ```-b``` followed by your own bucket name.

### 4. Create an empty database
The databse could be created either locally by the command below:
```
docker run --mount type=bind,source="$(pwd)"/data,target=/app/data runapp python3 src/models.py
```
If you want to create the database in AWS RDS, you need to modify the config/aws_rds.conf by replacing \<user> and \<password> by your own user and password.
and then type the command below:
```
docker run --mount type=bind,source="$(pwd)"/data,target=/app/data runapp python3 src/models.py --rds
```

