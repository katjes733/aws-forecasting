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
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import matplotlib.pyplot as plt
import numpy as np

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, NumeralTickFormatter, DatetimeTickFormatter
from bokeh.models.tools import HoverTool, CrosshairTool

import util.notebook_utils

default_df = '%Y%m%d_%H%M%S'
default_ui_df = '%a, %d %b %Y %H:%M:%S %Z'


def wait_till_delete(callback, check_time=5, timeout=None):

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

    raise TimeoutError("Forecast resource deletion timed-out.")


def wait(callback, time_interval=10):

    status_indicator = util.notebook_utils.StatusIndicator()

    while True:
        status = callback()['Status']
        status_indicator.update(status)
        if status in ('ACTIVE', 'CREATE_FAILED'):
            break
        time.sleep(time_interval)

    status_indicator.end()

    return (status == "ACTIVE")


def create_forecast_name(prefix, algorithm, date_format=default_df):
    forecast_name = f"{prefix}_{algorithm}"
    forecast_name_unique = f"{forecast_name}_{datetime.now().strftime(date_format)}"
    return forecast_name_unique


def create_forecast(forecast_name_unique, predictor_arn, forecast_client, existing_resource_strategy="Ask", tags=[], ui_date_format=default_ui_df):
    existing_forecasts = get_forecasts_by_predictor(predictor_arn, forecast_client, True)

    if len(existing_forecasts) > 0:
        print(f"Forecasts exist with the latest one from {existing_forecasts[0]['CreationTime'].strftime(ui_date_format)}.")
        if existing_resource_strategy == 'Keep':
            return existing_forecasts[0]["ForecastArn"]
        elif existing_resource_strategy == 'Ask':
            if not input("Create new forecast (y/N)? ").lower() == "y":
                return existing_forecasts[0]["ForecastArn"]

    create_forecast_response = forecast_client.create_forecast(ForecastName=forecast_name_unique,
                                                               PredictorArn=predictor_arn,
                                                               Tags=tags
                                                              )
    forecast_arn = create_forecast_response['ForecastArn']
    print(f"Creating new forecast {forecast_arn} for predictor {predictor_arn}...")
    status = util.wait(lambda: forecast_client.describe_forecast(ForecastArn=forecast_arn))
    assert status

    return forecast_arn


def get_forecasts(forecast_name, forecast_client, exact=True):
    args = {}
    forecasts = []
    while True:
        response = forecast_client.list_forecasts(**args)
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


def get_predictor(predictor_name, forecast_client, exact=True):
    predictors = get_predictors(predictor_name, forecast_client, exact)
    if len(predictors) == 0:
        return None
    else:
        return predictors[0]


def get_predictors(predictor_name, forecast_client, exact=True):
    args = {}
    predictors = []
    while True:
        response = forecast_client.list_predictors(**args)
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


def get_dataset_import_jobs(dataset_arn, forecast_client):
    dataset_name = re.search(r"^.*?dataset/(?P<dataset_name>[a-zA-Z0-9_]+)$", dataset_arn, re.M).group("dataset_name")

    args = {}
    dataset_import_jobs = []
    while True:
        response = forecast_client.list_dataset_import_jobs(**args)
        for dataset_import_job in response["DatasetImportJobs"]:
            dataset_name_from_arn = re.search("^.*?dataset-import-job/(?P<dataset_import_job_name>[a-zA-Z0-9_]+?)/.*$", dataset_import_job["DatasetImportJobArn"], re.M).group("dataset_import_job_name")
            if dataset_name_from_arn == dataset_name:
                dataset_import_jobs.append(dataset_import_job)
        if "NextToken" in response and response["NextToken"]:
            args = {"NextToken": response["NextToken"]}
        else:
            break
    dataset_import_jobs.sort(key=lambda f: f['CreationTime'], reverse=True)
    return dataset_import_jobs


def delete_predictor(predictor_arn: str, forecast_client):
    print(f"Deleting predictor {predictor_arn}...")
    wait_till_delete(lambda: forecast_client.delete_predictor(PredictorArn=predictor_arn))


def delete_predictors(dataset_group_arn: str, forecast_client):
    dataset_group_name = re.search(r"^.*?dataset-group/(?P<dataset_group_name>[a-zA-Z0-9_]+)$", dataset_group_arn, re.M).group("dataset_group_name")
    for predictor in get_predictors(dataset_group_name, forecast_client, exact=False):
        delete_forecasts_by_predictor(predictor["PredictorArn"], forecast_client)
        delete_predictor(predictor["PredictorArn"], forecast_client)


def delete_dataset_group(dataset_group_arn: str, forecast_client, s3_client=None):
    try:
        response = forecast_client.describe_dataset_group(DatasetGroupArn=dataset_group_arn)

        delete_predictors(dataset_group_arn, forecast_client)

        for dataset_arn in response["DatasetArns"]:
            delete_dataset(dataset_arn, forecast_client, s3_client)
        print(f"Deleting dataset_group {dataset_group_arn}...")
        wait_till_delete(lambda: forecast_client.delete_dataset_group(DatasetGroupArn=dataset_group_arn))
    except forecast_client.exceptions.ResourceNotFoundException:
        print(f"Dataset with ARN '{dataset_group_arn}' does not exist.")


def delete_dataset_import_jobs(dataset_arn, forecast_client, s3_client=None):
    for dataset_import_job in get_dataset_import_jobs(dataset_arn, forecast_client):
        print(f"Deleting dataset_import_job: {dataset_import_job['DatasetImportJobArn']}")
        wait_till_delete(lambda: forecast_client.delete_dataset_import_job(DatasetImportJobArn=dataset_import_job["DatasetImportJobArn"]))
        if s3_client:
            path = dataset_import_job['DataSource']['S3Config']['Path']
            print(f"Deleting file: {path}")
            regex_result = re.search(r"s3://(?P<bucket>[a-z0-9-]+)/(?P<key>.+)", path, re.M)
            bucket = regex_result.group('bucket')
            key = regex_result.group('key')
            s3_client.delete_object(Bucket=bucket, Key=key)
        else:
            print("S3 Client not defined")


def delete_dataset(arn, forecast_client, s3_Client=None):
    delete_dataset_import_jobs(arn, forecast_client, s3_Client)
    print(f"Deleting dataset: {arn}")
    util.wait_till_delete(lambda: forecast_client.delete_dataset(DatasetArn=arn))


def get_dataframe(filename, initial_timestamp_column_name, column_names_map, target_column_name):
    if initial_timestamp_column_name not in column_names_map.keys():
        raise KeyError(f"{filename}: Column {initial_timestamp_column_name} does not exist in column names dictionary keys: {column_names_map.keys()}")

    if target_column_name not in column_names_map.values():
        raise KeyError(f"{filename}: Column {target_column_name} does not exist in column names dictionary values: {column_names_map.values()}")

    try:
        df = pd.read_csv(filename, dtype=object, thousands=',')

        if initial_timestamp_column_name not in df.columns:
            raise KeyError(f"{filename}: Column {initial_timestamp_column_name} does not exist in dataframe columns: {df.columns}")

        initial_target_column_name = list(column_names_map.keys())[list(column_names_map.values()).index(target_column_name)]
        if initial_target_column_name not in df.columns:
            raise KeyError(f"{filename}: Column {initial_target_column_name} does not exist in dataframe columns: {df.columns}")
    except Exception as e:
        if type(e).__name__ == "FileNotFoundError":
            df = pd.DataFrame()
        else:
            print(type(e).__name__)
            raise e

    return df


def prepare_data(bucket_name,
                 data_key,
                 date_format,
                 initial_timestamp_column_name: str,
                 column_names_map: dict[str, str],
                 target_column_name,
                 item_id,
                 fill_missing_values=False,
                 use_bank_day=False,
                 minimal=False,
                 start_date=None,
                 end_date=None):
    df = get_dataframe(filename=f"s3://{bucket_name}/{data_key}",
                       initial_timestamp_column_name=initial_timestamp_column_name,
                       column_names_map=column_names_map,
                       target_column_name=target_column_name
                       )
    
    if df.empty:
        return df
    
    col_to_drop = list(filter(lambda col: col not in column_names_map.keys(), df.columns))
    if len(col_to_drop) > 0:
        df.drop(col_to_drop, axis=1, inplace=True)

    # Drop columns with no values
    df.dropna(how='all', axis=1, inplace=True)

    if initial_timestamp_column_name in column_names_map.keys():
        df[initial_timestamp_column_name] = pd.to_datetime(df[initial_timestamp_column_name], format=date_format).dt.date
        if start_date and end_date:
            mask = (df[initial_timestamp_column_name] >= (start_date.to_pydatetime().date() if fill_missing_values else start_date)) & (df[initial_timestamp_column_name] <= (end_date.to_pydatetime().date() if fill_missing_values else end_date))
            df = df.loc[mask]
        elif start_date and not end_date:
            mask = (df[initial_timestamp_column_name] >= start_date.to_pydatetime().date())
            df = df.loc[mask]
        elif not start_date and end_date:
            mask = (df[initial_timestamp_column_name] <= end_date.to_pydatetime().date())
            df = df.loc[mask]

    col_as_float = list(filter(lambda col: col != initial_timestamp_column_name, df.columns))
    if len(col_as_float) > 0:
        try:
            df = to_numeric_df(df, col_as_float)
        except Exception as e:
            print(f"s3://{bucket_name}/{data_key} has NaN in columns {col_as_float}.")
            raise e

    df.rename(columns=column_names_map, inplace=True)
    df["item_id"] = item_id

    if use_bank_day:
        df["bank_day"] = 1

    if fill_missing_values:
        new_index = pd.Index(pd.date_range(df.timestamp.min(), df.timestamp.max()), name=column_names_map[initial_timestamp_column_name])
        df.set_index(column_names_map[initial_timestamp_column_name]).reindex(new_index)
        df = df.set_index(column_names_map[initial_timestamp_column_name]).reindex(new_index).reset_index()
        for col in df.columns:
            if use_bank_day and col == "bank_day":
                df.fillna({col: 0}, inplace=True)
            else:
                df[col] = df[col].ffill()
        df = df.sort_values(by=[column_names_map[initial_timestamp_column_name]], ascending=False, ignore_index=True)

    if use_bank_day:
        df["bank_day"] = pd.to_numeric(df["bank_day"]).astype("Int32")

    if minimal:
        col_to_drop = list(filter(lambda col: col not in [column_names_map[initial_timestamp_column_name], "item_id", target_column_name], df.columns))
        df.drop(col_to_drop, axis=1, inplace=True)

    return df


def is_excluded(obj: str, exclude: list[str]=[]):
    if not exclude:
        return False

    for one_exclude in exclude:
        if one_exclude in obj:
            return True

    return False


def get_related_data(s3_client,
                     bucket: str,
                     prefix: str,
                     target_df,
                     initial_timestamp_column_name: str,
                     column_names_map: dict[str, str],
                     target_column_name: str,
                     item_id: str,
                     exclude: list[str]=[],
                     fill_missing_values=False,
                     use_bank_day=False,
                     extra_features=True,
                     start_date=None,
                     end_date=None):
    object_prefix = prefix if prefix.endswith('/') else f"{prefix}/"

    args = {
        "Bucket": bucket,
        "Prefix": object_prefix
    }
    related_data_csv_objects = {}
    while True:
        response = s3_client.list_objects_v2(**args)
        list_related_data_csv_objects = list(filter(lambda obj: obj != object_prefix and not is_excluded(obj, exclude), list(map(lambda obj: obj["Key"], response["Contents"]))))
        related_data_csv_objects = { Path(obj).stem.lower(): obj for obj in list_related_data_csv_objects }
        if "NextContinuationToken" in response:
            args["ContinuationToken"] = response["NextContinuationToken"]
        else:
            break

    if extra_features:
        relevant_target_df_columns = list(filter(lambda col: col != target_column_name, target_df.columns))
    else:
        relevant_target_df_columns = [column_names_map[initial_timestamp_column_name], "item_id"]
        if use_bank_day:
            relevant_target_df_columns.append("bank_day")

    related_stocks_merged_df = target_df[relevant_target_df_columns].set_index(column_names_map[initial_timestamp_column_name])

    for key in related_data_csv_objects:
        related_stock_df = prepare_data(
            bucket,
            related_data_csv_objects[key],
            "%m/%d/%Y",
            initial_timestamp_column_name,
            column_names_map,
            target_column_name,
            key,
            fill_missing_values=fill_missing_values,
            use_bank_day=use_bank_day,
            start_date=start_date,
            end_date=end_date
        )[[column_names_map[initial_timestamp_column_name], target_column_name]].rename(columns={target_column_name: f"{key}_{target_column_name}"}).set_index(column_names_map[initial_timestamp_column_name])

        related_stocks_merged_df = related_stocks_merged_df.join(related_stock_df)

    return related_stocks_merged_df.reset_index()


def load_exact_sol(fname, item_id, is_schema_perm=False, target_col_name='target'):
    exact = pd.read_csv(fname, header=None, thousands=',')
    exact.columns = ['timestamp', 'item_id', target_col_name]
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
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(assume_role_policy_document)
        )
        role_arn = create_role_response["Role"]["Arn"]
        print("Created", role_arn)

        print("Attaching policies...")
        iam.attach_role_policy(
            RoleName=role_name,
            PolicyArn="arn:aws:iam::aws:policy/AmazonForecastFullAccess"
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


def delete_iam_role(role_name):
    iam = boto3.client("iam")
    iam.detach_role_policy(PolicyArn="arn:aws:iam::aws:policy/AmazonS3FullAccess", RoleName=role_name)
    iam.detach_role_policy(PolicyArn="arn:aws:iam::aws:policy/AmazonForecastFullAccess", RoleName=role_name)
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


def plot_bokeh_forecasts(forecast_dfs, exact, freq='1D', forecastHorizon=30, time_back=30, future=pd.DataFrame(), target_col_name='target', reverse=False, title="Stock Price Forecast"):
    p10 = forecast_dfs['p10']
    p50 = forecast_dfs['p50']
    p90 = forecast_dfs['p90']
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
    plot = figure(title=title, x_axis_label='Timestamp', y_axis_label='Stock Value', sizing_mode='stretch_width')

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


def extract_gz(src, dst):

    print(f"Extracting {src} to {dst}")

    with open(dst, 'wb') as fd_dst:
        with gzip.GzipFile(src, 'rb') as fd_src:
            data = fd_src.read()
            fd_dst.write(data)

    print("Done.")


def get_attributes_by_domain(domain: str):
    match domain:
        case "RETAIL":
            return {"AttributeName": "demand", "AttributeType": "float"}
        case "CUSTOM":
            return {"AttributeName": "target_value", "AttributeType": "integer"}
        case "INVENTORY_PLANNING":
            return {"AttributeName": "demand", "AttributeType": "float"}
        case "EC2_CAPACITY":
            return {"AttributeName": "number_of_instances", "AttributeType": "integer"}
        case "WORK_FORCE":
            return {"AttributeName": "workforce_demand", "AttributeType": "integer"}
        case "WEB_TRAFFIC":
            return {"AttributeName": "value", "AttributeType": "float"}
        case "METRICS":
            return {"AttributeName": "metric_value", "AttributeType": "integer"}
        case _:
            raise ValueError(f"domain must be 'RETAIL'|'CUSTOM'|'INVENTORY_PLANNING'|'EC2_CAPACITY'|'WORK_FORCE'|'WEB_TRAFFIC'|'METRICS', but was {domain}")


def get_schema_attributes(df, domain: str, target_column_name: str):
    attributes = []
    for one_column in df:
        if one_column == target_column_name:
            attributes.append(get_attributes_by_domain(domain))
        elif one_column == "item_id":
            attributes.append({"AttributeName": one_column, "AttributeType": "string"})
        elif one_column == "timestamp":
            attributes.append({"AttributeName": one_column, "AttributeType": "timestamp"})
        elif one_column == "bank_day":
            attributes.append({"AttributeName": one_column, "AttributeType": "integer"})
        else:
            attributes.append({"AttributeName": f"{one_column}_value", "AttributeType": "float"})
    return attributes


def get_relevant_forecasts(forecast_name_prefix: str, versions: list[int], algorithm: str, forecast_client):
    relevant_forecasts = {}
    for version in versions:
        forecasts = util.get_forecasts_by_predictor(f"{forecast_name_prefix}{version}_{algorithm}".replace("-", "_"), forecast_client, exact=False)
        if len(forecasts) > 0:
            relevant_forecasts[str(version)] = forecasts[0]["ForecastArn"]
    return relevant_forecasts


def query_forecasts(forecasts, item_id: str, forecastquery_client):
    query_results = {}
    for key in forecasts:
        forecast_response = forecastquery_client.query_forecast(
            ForecastArn=forecasts[key],
            Filters={"item_id": item_id})
        query_results[key] = forecast_response["Forecast"]
    return query_results


def query_results_to_dataframes(query_results, fill_missing_values=False):
    results = {}
    for version in query_results:
        dataframes = {}
        query_result = query_results[version]["Predictions"]
        for key in query_result:
            df = pd.DataFrame(query_result[key])
            df["Timestamp"] = pd.to_datetime(df["Timestamp"])
            if not fill_missing_values:
                holidays = calendar().holidays(start=df["Timestamp"].min(), end=df["Timestamp"].max())
                df = df[(df["Timestamp"].dt.dayofweek < 5) & (~df["Timestamp"].isin(holidays))]
            dataframes[key] = df
        results[version] = dataframes
    return results


def to_numeric(value: str):
    regex_result = re.search(r"^(?P<number>[0-9,.]*)(?P<factor>[TMB]?)$", value, re.M)

    if not regex_result:
        return np.nan
    else:
        numeric_value = pd.to_numeric(regex_result.group("number").replace(',', ''), errors='coerce')
        if pd.isna(numeric_value):
            return numeric_value
        if not regex_result.group("factor"):
            return numeric_value
        elif regex_result.group("factor") == "T":
            return numeric_value * 1_000
        elif regex_result.group("factor") == "M":
            return numeric_value * 1_000_000
        elif regex_result.group("factor") == "B":
            return numeric_value * 1_000_000_000


def to_numeric_df(df, columns: list[str]):
    if not columns or len(columns) == 0:
        return df
    else:
        for col in df.columns:
            if col in columns:
                df[col] = df[col].apply(to_numeric)
        return df


def extract_summary_metrics(metric_response, predictor_name):
    df = pd.DataFrame(metric_response['PredictorEvaluationResults'][0]['TestWindows'][0]['Metrics']['WeightedQuantileLosses'])
    df['Predictor'] = predictor_name
    return df


def get_exact_by_forecast(forecast_arn: str, item_id: str, target_col_name: str, forecast_client):
    dataset_group_arn = forecast_client.describe_forecast(ForecastArn=forecast_arn)['DatasetGroupArn']

    dataset_arn_tts = list(filter(lambda ds: ds.endswith('_tts'), forecast_client.describe_dataset_group(DatasetGroupArn=dataset_group_arn)['DatasetArns']))[0]
    exact_path = util.get_dataset_import_jobs(dataset_arn_tts, forecast_client)[0]['DataSource']['S3Config']['Path']

    exact_df = util.load_exact_sol(exact_path, item_id, target_col_name=target_col_name)

    return exact_df


def get_exacts_by_forecasts(forecast_arns, item_id: str, target_col_name: str, forecast_client):
    exact_dfs = {}
    for version, forecast_arn in forecast_arns.items():
        exact_df = get_exact_by_forecast(forecast_arn, item_id, target_col_name, forecast_client)
        exact_dfs[version] = exact_df

    return exact_dfs


def get_simple_tags_by_arn(arn: str, forecast_client):
    return get_simple_tags_by_tags(forecast_client.list_tags_for_resource(ResourceArn=arn)['Tags'])


def get_simple_tags_by_tags(resource_tags: list[dict]):
    simple_tags = {}
    for resource_tag in resource_tags:
        simple_tags[resource_tag['Key']] = resource_tag['Value']
    return simple_tags


def get_versions(max_value: int, patterns=None):
    MIN = 1
    individual_versions = []
    patterns_stripped = patterns.strip() if patterns else patterns
    patterns_edited = patterns_stripped if patterns_stripped else f"{MIN}-{max_value}"
    for one_pattern in patterns_edited.split(','):
        single = None
        start = None
        end = None
        one_pattern_stripped = one_pattern.strip()
        match = re.search(r"^(?P<start>\d*)\-(?P<end>\d*)$|^(?P<single>\d+)$", one_pattern_stripped, re.M)
        if match and match.group('single'):
            single = match.group('single')
        if match and match.group('start'):
            start = match.group('start')
        if match and match.group('end'):
            end = match.group('end')

        if start and end and not single:
            individual_versions.extend(list(range(int(start), int(end) + 1)))
        elif start and not end and not single:
            individual_versions.extend(list(range(int(start), max_value + 1)))
        elif not start and end and not single:
            individual_versions.extend(list(range(MIN, int(end) + 1)))
        elif not start and not end and single:
            individual_versions.append(int(single))

    individual_versions.sort()
    return list(filter(lambda version: version >= MIN and version <= max_value, individual_versions))


def is_recreate_resource(
        resource_arn: str,
        forecast_client,
        date_format: str,
        min_date_df,
        max_date_df,
        existing_resource_strategy: str,
        user_question="Recreate resource"):
    all_tags = util.get_simple_tags_by_arn(resource_arn, forecast_client)

    force_recreate = False
    if 'DATE_RANGE_MIN' in all_tags and all_tags['DATE_RANGE_MIN'] and \
            'DATE_RANGE_MAX' in all_tags and all_tags['DATE_RANGE_MAX']:
        min_date_tag = datetime.strptime(all_tags['DATE_RANGE_MIN'], date_format).date()
        max_date_tag = datetime.strptime(all_tags['DATE_RANGE_MAX'], date_format).date()

        if min_date_tag == min_date_df and max_date_tag == max_date_df:
            print(f'Date ranges for dataframe ({min_date_df} - {max_date_df}) and from tags ({min_date_tag} - {max_date_tag}) match.')
        else:
            print(f'Date ranges for dataframe ({min_date_df} - {max_date_df}) and from tags ({min_date_tag} - {max_date_tag}) do not match.')
            force_recreate = True
    else:
        print('Date range tags are incomplete.')
        force_recreate = True

    if existing_resource_strategy == 'Keep' and not force_recreate:
        return False
    elif existing_resource_strategy == 'Recreate' or (existing_resource_strategy == 'Keep' and force_recreate):
        return True
    elif existing_resource_strategy == 'Ask':
        if input(f"{user_question.strip()} (y/N)? ").lower() == "y":
            return True
        else:
            return False
    else:
        raise ValueError("Cannot determine whether or not to recreate resource")