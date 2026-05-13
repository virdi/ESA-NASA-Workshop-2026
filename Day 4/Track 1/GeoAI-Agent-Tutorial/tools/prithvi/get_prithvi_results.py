import json
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError


def get_prithvi_results(job_id: str) -> dict:
    """Retrieve outputs for a finished Prithvi inference job.

    job_id is the S3 OutputLocation returned by run_prithvi_inference.
    Returns result_urls (GeoTIFFs) and summary statistics per output.md schema.
    Call only after get_prithvi_job_status returns 'finished'.
    """
    parsed = urlparse(job_id)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")

    s3 = boto3.client("s3")
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        return json.loads(response["Body"].read())
    except ClientError as e:
        return {"message": f"Results retrieval failed: {e}"}
