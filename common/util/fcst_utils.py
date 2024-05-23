import time
import json
import gzip
import re
import math
from datetime import datetime
from pathlib import Path

import boto3
import botocore.exceptions

import pandas as pd
import matplotlib.pyplot as plt

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, NumeralTickFormatter, DatetimeTickFormatter
from bokeh.models.tools import HoverTool, CrosshairTool

import util.notebook_utils

default_df = '%Y%m%d_%H%M%S'
default_ui_df = '%a, %d %b %Y %H:%M:%S %Z'


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


def create_forecast_name(prefix, algorithm, date_format=default_df):
    forecast_name = f"{prefix}_{algorithm}"
    forecast_name_unique = f"{forecast_name}_{datetime.now().strftime(date_format)}"
    return forecast_name, forecast_name_unique


def create_forecast(forecast_name, forecast_name_unique, predictor_arn, forecast_client, tags=[], ui_date_format=default_ui_df):
    existing_forecasts = util.get_forecasts(forecast_name, forecast_client, False)
    
    if len(existing_forecasts) > 0:
        print(f"Forecasts exist with the latest one from {existing_forecasts[0]['CreationTime'].strftime(ui_date_format)}.")
        if not input("Create new forecast (y/N)? ").lower() == "y":
            return existing_forecasts[0]["ForecastArn"]

    create_forecast_response = forecast_client.create_forecast(ForecastName=forecast_name_unique,
                                                               PredictorArn=predictor_arn,
                                                               Tags=tags
                                                              )
    forecast_arn = create_forecast_response['ForecastArn']
    print(f"Creating new forecast {forecast_arn} for predictor {predictor_arn}...")
    status = util.wait(lambda: forecast.describe_forecast(ForecastArn=forecast_arn))
    assert status
    
    return forecast_arn


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


def prepare_data(bucket_name, data_key, date_format, target_column_name, item_id, fill_missing_values=False, minimal=False, start_date=None, end_date=None):
    df = pd.read_csv(f"s3://{bucket_name}/{data_key}", dtype=object)
    df.drop(["Vol.", "Change %"], axis=1, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"], format=date_format).dt.date
    df[["Price", "Open", "High", "Low"]] = df[["Price", "Open", "High", "Low"]].astype(float)
    df.rename(columns={'Date': 'datetime', 'Price': target_column_name, 'Open': 'open', 'High': 'high', 'Low': 'low'}, inplace=True)
    df["item_id"] = item_id
    
    if start_date and end_date:
        mask = (df['datetime'] >= start_date.to_pydatetime().date()) & (df['datetime'] <= end_date.to_pydatetime().date())
        df = df.loc[mask]
    elif start_date and not end_date:
        mask = (df['datetime'] >= start_date.to_pydatetime().date())
        df = df.loc[mask]
    elif not start_date and end_date:
        mask = (df['datetime'] <= end_date.to_pydatetime().date())
        df = df.loc[mask]

    if fill_missing_values:
        new_index = pd.Index(pd.date_range(df.datetime.min(), df.datetime.max()), name="datetime")
        df.set_index("datetime").reindex(new_index)
        df = df.set_index("datetime").reindex(new_index).reset_index().ffill()
        df = df.sort_values(by=['datetime'], ascending=False, ignore_index=True)

    if minimal:
        df.drop(["open", "high", "low"], axis=1, inplace=True)
        df.rename(columns={'datetime': 'timestamp'}, inplace=True)

    return df


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


def plot_forecasts(fcsts, exact, freq = '1D', forecastHorizon=30, time_back = 30, future=pd.DataFrame(), target_col_name='target', reverse=False):
    p10 = pd.DataFrame(fcsts['Forecast']['Predictions']['p10'])
    p50 = pd.DataFrame(fcsts['Forecast']['Predictions']['p50'])
    p90 = pd.DataFrame(fcsts['Forecast']['Predictions']['p90'])
    pred_int = p50['Timestamp'].apply(lambda x: pd.Timestamp(x))
    p50['Timestamp'] = pred_int
    fcst_start_date = pred_int.iloc[0]
    fcst_end_date = pred_int.iloc[-1]
    tb = exact.head(time_back) if reverse else exact.tail(time_back)
    tb.drop(["item_id"], axis=1, inplace=True)
    tb.rename(columns={'timestamp': 'Timestamp', target_col_name: 'Value'}, inplace=True)
    time_int = tb['Timestamp'].apply(lambda x: pd.Timestamp(x))
    tb['Timestamp'] = time_int
    final = pd.concat([p50, tb], ignore_index=True).sort_values('Timestamp', ignore_index=True)
    plt.plot(final['Timestamp'].values, final['Value'].values, color = 'k')
    plt.fill_between(pred_int,
                     p10['Value'].values,
                     p90['Value'].values,
                     color='b', alpha=0.3);
    plt.axvline(x=pd.Timestamp(fcst_start_date), linewidth=1, color='g', ls='dashed')
    plt.axvline(x=pd.Timestamp(fcst_end_date), linewidth=1, color='g', ls='dashed')
    plt.xticks(rotation=30)
    plt.legend(['Target', 'Forecast'], loc = 'lower left')

    if not future.empty:
        future_df = future.rename(columns={'timestamp': 'Timestamp', target_col_name: 'Value'})
        plt.plot(future_df['Timestamp'].values, future_df['Value'].values, color = 'r')


def plot_bokeh_forecasts(fcsts, exact, freq = '1D', forecastHorizon=30, time_back = 30, future=pd.DataFrame(), target_col_name='target', reverse=False):
    p10 = pd.DataFrame(fcsts['Forecast']['Predictions']['p10'])
    p50 = pd.DataFrame(fcsts['Forecast']['Predictions']['p50'])
    p90 = pd.DataFrame(fcsts['Forecast']['Predictions']['p90'])
    pred_int = p50['Timestamp'].apply(lambda x: pd.Timestamp(x))
    p50['Timestamp'] = pred_int
    fcst_start_date = pred_int.iloc[0]
    fcst_end_date = pred_int.iloc[-1]
    tb = exact.head(time_back) if reverse else exact.tail(time_back)
    tb.drop(["item_id"], axis=1, inplace=True)
    tb.rename(columns={'timestamp': 'Timestamp', target_col_name: 'Value'}, inplace=True)
    time_int = tb['Timestamp'].apply(lambda x: pd.Timestamp(x))
    tb['Timestamp'] = time_int
    final = pd.concat([p50, tb], ignore_index=True).sort_values('Timestamp', ignore_index=True)

    hover = HoverTool(tooltips=[('Timestamp', '@Timestamp{%F}'), ('Value', '$@Value{0,0.00}')],
                      formatters={'@Timestamp': 'datetime'},
                      mode='vline'
                     )
    crosshair = CrosshairTool()
    tools = (hover, crosshair)

    source = ColumnDataSource(final)
    plot = figure(title="Stock Price Forecast", x_axis_label='Timestamp', y_axis_label='Stock Value', sizing_mode='stretch_width')

    main = plot.line(x='Timestamp', y='Value', source=source, line_width=2)
    plot.add_tools(*tools)
    plot.yaxis[0].formatter = NumeralTickFormatter(format="$0.00")
    plot.xaxis[0].formatter = DatetimeTickFormatter(hours="%d %b %Y",
                                                 days="%d %b %Y",
                                                 months="%d %b %Y",
                                                 years="%d %b %Y",
                                                )
    plot.xaxis[0].major_label_orientation = math.pi/3
    plot.varea(pred_int, p10['Value'].values, p90['Value'].values, fill_color=("blue"), fill_alpha=0.3)
    plot.vspan(x=pd.Timestamp(fcst_start_date), line_color="green", line_width=3, line_dash='dashed')
    plot.vspan(x=pd.Timestamp(fcst_end_date), line_color="green", line_width=3, line_dash='dashed')
    hover.renderers = [main]

    if not future.empty:
        future_source = ColumnDataSource(future.rename(columns={'timestamp': 'Timestamp', target_col_name: 'Value'}))
        future_plot = plot.line(x='Timestamp', y='Value', source=future_source, line_width=2, line_color='red')
        hover.renderers.append(future_plot)

    return plot


def extract_gz( src, dst ):
    
    print( f"Extracting {src} to {dst}" )    

    with open(dst, 'wb') as fd_dst:
        with gzip.GzipFile( src, 'rb') as fd_src:
            data = fd_src.read()
            fd_dst.write(data)

    print("Done.")
