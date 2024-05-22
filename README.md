# AWS Forecasting

A little forecasting project to apply forecasts to stock prices.

## Summary

This is a simple use case for usage of AWS Forecast to forecast stock prices. There is no garantuee for forecasts to be accurate. It is merely a proof of concept to explore various ways to influence forecast results and achieve results that resemble real stock prices as accurately as possible.

At the current stage, we are looking a historical values for a single stock only and attempt to predict future values based on this single stock only. Obviously, this is far from optimal, but in later iterations, we will incorporate usage of historical peer stock values, indices and possibly other events to forecast a single stock value.

All steps are manual through a Jupyter notebook. There is no on-click automation in place yet.

## Prerequisites

* SageMaker role already has all necessary permissions to invoke various Forecast APIs
* S3 bucket exists to provide data to SageMaker
* Historical stock prices are available as CSV
* Variables in Step 1.2 are customized to your individual needs