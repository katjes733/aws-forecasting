import time
import json
import gzip
import re

import boto3
import botocore.exceptions

import pandas as pd
import matplotlib.pyplot as plt

import util.notebook_utils


def wait_till_delete(callback, check_time = 5, timeout = None):

    elapsed_time = 0
    while timeout is None or elapsed_time < timeout:
        try:
            out = callback()
        except botocore.exceptions.ClientError as e:
            # When given the resource not found exception, deletion has occured
            if e.response['Error']['Code'] == 'ResourceNotFoundException':
                print('Successful delete')
                return
            else:
                raise
        time.sleep(check_time)  # units of seconds
        elapsed_time += check_time

    raise TimeoutError( "Forecast resource deletion timed-out." )


def wait(callback, time_interval = 10):

    status_indicator = util.notebook_utils.StatusIndicator()

    while True:
        status = callback()['Status']
        status_indicator.update(status)
        if status in ('ACTIVE', 'CREATE_FAILED'): break
        time.sleep(time_interval)

    status_indicator.end()

    return (status == "ACTIVE")


def get_forecasts(forecast_name, forecast, exact=True):
    args = {}
    forecasts = []
    while True:
        response = forecast.list_forecasts(**args)
        for one_forecast in response["Forecasts"]:
            if exact and one_forecast["ForecastName"] == forecast_name:
                forecasts.append(one_forecast)
            elif not exact and forecast_name in one_forecast["ForecastName"]:
                forecasts.append(one_forecast)
        if "NextToken" in response and response["NextToken"]:
            args = {"NextToken": response["NextToken"]}
        else:
            break

    return sorted(forecasts, reverse=True)


def get_forecasts_by_predictor(predictor_arn, forecast_client, exact=True):
    args = {}
    forecasts = []
    while True:
        response = forecast_client.list_forecasts(**args)
        for one_forecast in response["Forecasts"]:
            if exact and one_forecast["PredictorArn"] == predictor_arn:
                forecasts.append(one_forecast)
            elif not exact and predictor_arn in one_forecast["PredictorArn"]:
                forecasts.append(one_forecast)
        if "NextToken" in response and response["NextToken"]:
            args = {"NextToken": response["NextToken"]}
        else:
            break
    forecasts.sort(key=lambda f: f['CreationTime'], reverse=True)
    return forecasts


def delete_forecasts_by_predictor(predictor_arn, forecast_client, exact=True):
    for one_forecast in get_forecasts_by_predictor(predictor_arn, forecast_client, exact):
        print(f"Deleting forecast {one_forecast['ForecastArn']}...")
        util.wait_till_delete(lambda: forecast_client.delete_forecast(ForecastArn=one_forecast["ForecastArn"]))


def get_predictor(predictor_name, forecast, exact=True):
    predictors = get_predictors(predictor_name, forecast, exact)
    if len(predictors) == 0:
        return None
    else:
        return predictors[0]


def get_predictors(predictor_name, forecast, exact=True):
    args = {}
    predictors = []
    while True:
        response = forecast.list_predictors(**args)
        for one_predictor in response["Predictors"]:
            if exact and one_predictor["PredictorName"] == predictor_name:
                predictors.append(one_predictor)
            elif not exact and predictor_name in one_predictor["PredictorName"]:
                predictors.append(one_predictor)
        if "NextToken" in response and response["NextToken"]:
            args = {"NextToken": response["NextToken"]}
        else:
            break
    predictors.sort(key=lambda p: p['CreationTime'], reverse=True)
    return predictors


def get_dataset_import_jobs(dataset_arn, forecast):
    dataset_name = re.search(r"^.*?dataset/(?P<dataset_name>.*)$", dataset_arn, re.M).group("dataset_name")

    args = {}
    dataset_import_jobs = []
    while True:
        response = forecast.list_dataset_import_jobs(**args)
        for dataset_import_job in response["DatasetImportJobs"]:
            dataset_name_from_arn = re.search("^.*?dataset-import-job/(?P<dataset_import_job_name>.*?)/.*$", dataset_import_job["DatasetImportJobArn"], re.M).group("dataset_import_job_name")
            if dataset_name_from_arn == dataset_name:
                dataset_import_jobs.append(dataset_import_job)
        if "NextToken" in response and response["NextToken"]:
            args = {"NextToken": response["NextToken"]}
        else:
            break
    return dataset_import_jobs


def delete_predictors():
    None


def delete_dataset_group(arn, forecast):
    try:
        response = forecast.describe_dataset_group(DatasetGroupArn=arn)
        for dataset_arn in response["DatasetArns"]:
            delete_dataset(dataset_arn, forecast)
        print(f"Deleting dataset_group: {arn}")
        wait_till_delete(lambda: forecast.delete_dataset_group(DatasetGroupArn=arn))
    except forecast.exceptions.ResourceNotFoundException:
        print(f"Dataset with ARN '{arn}' does not exist.")


def delete_dataset_import_jobs(dataset_arn, forecast):
    for dataset_import_job in get_dataset_import_jobs(dataset_arn, forecast):
        print(f"Deleting dataset_import_job: {dataset_import_job['DatasetImportJobArn']}")
        wait_till_delete(lambda: forecast.delete_dataset_import_job(DatasetImportJobArn=dataset_import_job["DatasetImportJobArn"]))


def delete_dataset(arn, forecast):
    delete_dataset_import_jobs(arn, forecast)
    print(f"Deleting dataset: {arn}")
    util.wait_till_delete(lambda: forecast.delete_dataset(DatasetArn=arn))


def load_exact_sol(fname, item_id, is_schema_perm=False, target_col_name='target'):
    exact = pd.read_csv(fname, header=None)
    exact.columns = ['item_id', 'timestamp', target_col_name]
    if is_schema_perm:
        exact.columns = ['timestamp', target_col_name, 'item_id']
    return exact.loc[exact['item_id'] == item_id]


def get_or_create_iam_role( role_name ):

    iam = boto3.client("iam")

    assume_role_policy_document = {
        "Version": "2012-10-17",
        "Statement": [
            {
              "Effect": "Allow",
              "Principal": {
                "Service": "forecast.amazonaws.com"
              },
              "Action": "sts:AssumeRole"
            }
        ]
    }

    try:
        create_role_response = iam.create_role(
            RoleName = role_name,
            AssumeRolePolicyDocument = json.dumps(assume_role_policy_document)
        )
        role_arn = create_role_response["Role"]["Arn"]
        print("Created", role_arn)
        
        print("Attaching policies...")
        iam.attach_role_policy(
            RoleName = role_name,
            PolicyArn = "arn:aws:iam::aws:policy/AmazonForecastFullAccess"
        )

        iam.attach_role_policy(
            RoleName=role_name,
            PolicyArn='arn:aws:iam::aws:policy/AmazonS3FullAccess',
        )

        print("Waiting for a minute to allow IAM role policy attachment to propagate...")
        time.sleep(60)
    except iam.exceptions.EntityAlreadyExistsException:
        print("The role " + role_name + " exists, skipping creation...")
        role_arn = boto3.resource('iam').Role(role_name).arn

    print("Done.")
    return role_arn


def delete_iam_role( role_name ):
    iam = boto3.client("iam")
    iam.detach_role_policy( PolicyArn = "arn:aws:iam::aws:policy/AmazonS3FullAccess", RoleName = role_name )
    iam.detach_role_policy( PolicyArn = "arn:aws:iam::aws:policy/AmazonForecastFullAccess", RoleName = role_name )
    iam.delete_role(RoleName=role_name)


def get_or_create_bucket(bucket_name, region=None):
    """Create an S3 bucket in a specified region
    If a region is not specified, the bucket is created in the S3 default
    region (us-east-1).
    :param bucket_name: Bucket to create
    :param region: String region to create bucket in, e.g., 'us-west-2'
    :return: True if bucket created, else False
    """
    account_id = boto3.client('sts').get_caller_identity().get('Account')

    if region is None:
        region = "us-east-1"

    if account_id not in bucket_name and region not in bucket_name:
        bucket_name = f"{bucket_name}-{account_id}-{region}"

    try:
        if region == "us-east-1":
            s3_client = boto3.client('s3')
            s3_client.create_bucket(Bucket=bucket_name)
        else:
            s3_client = boto3.client('s3', region_name=region)
            location = {'LocationConstraint': region}
            s3_client.create_bucket(Bucket=bucket_name,
                                    CreateBucketConfiguration=location)
    except Exception as e:
        print(e)

    return bucket_name


def plot_forecasts(fcsts, exact, freq = '1D', forecastHorizon=30, time_back = 30, target_col_name='target', reverse=False):
    p10 = pd.DataFrame(fcsts['Forecast']['Predictions']['p10'])
    p50 = pd.DataFrame(fcsts['Forecast']['Predictions']['p50'])
    p90 = pd.DataFrame(fcsts['Forecast']['Predictions']['p90'])
    pred_int = p50['Timestamp'].apply(lambda x: pd.Timestamp(x))
    fcst_start_date = pred_int.iloc[0]
    fcst_end_date = pred_int.iloc[-1]
    time_int = exact['timestamp'].apply(lambda x: pd.Timestamp(x))
    plt.plot(time_int[(time_back if reverse else -time_back):],exact[target_col_name].values[(time_back if reverse else -time_back):], color = 'r')
    plt.plot(pred_int, p50['Value'].values, color = 'k')
    plt.fill_between(pred_int, 
                     p10['Value'].values,
                     p90['Value'].values,
                     color='b', alpha=0.3)
    plt.axvline(x=pd.Timestamp(fcst_start_date), linewidth=3, color='g', ls='dashed')
    plt.axvline(x=pd.Timestamp(fcst_end_date), linewidth=3, color='g', ls='dashed')
    plt.xticks(rotation=30)
    plt.legend(['Target', 'Forecast'], loc = 'lower left')


def extract_gz( src, dst ):
    
    print( f"Extracting {src} to {dst}" )    

    with open(dst, 'wb') as fd_dst:
        with gzip.GzipFile( src, 'rb') as fd_src:
            data = fd_src.read()
            fd_dst.write(data)

    print("Done.")
