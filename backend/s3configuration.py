import boto3
import os
from botocore.exceptions import ClientError

s3_client = boto3.client("s3", aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID"), aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY"), region_name = os.getenv("AWS_REGION"))
bucket_name = os.getenv("AWS_BUCKET_NAME")

def upload_to_s3(file_path, s3_key):
    pass


    