import boto3
from botocore import UNSIGNED
from botocore.client import Config
import os
import time

s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED), region_name="us-east-1")

bucket = "aft-vbi-pds"
img_prefix = "bin-images/"
meta_prefix = "metadata/"

os.makedirs("bin-images", exist_ok=True)
os.makedirs("metadata", exist_ok=True)

continuation_token = None
count = 0
max_download = 5000  # change to None for full dataset

while True:
    if continuation_token:
        response = s3.list_objects_v2(
            Bucket=bucket, Prefix=img_prefix, ContinuationToken=continuation_token, MaxKeys=1000
        )
    else:
        response = s3.list_objects_v2(
            Bucket=bucket, Prefix=img_prefix, MaxKeys=1000
        )

    for obj in response.get("Contents", []):
        key = obj["Key"]
        filename = key.split("/")[-1]

        if not filename.endswith(".jpg"):
            continue

        # Skip existing files to resume interrupted downloads
        if os.path.exists(f"bin-images/{filename}"):
            continue

        try:
            # Download image
            s3.download_file(bucket, key, f"bin-images/{filename}")
            # Download corresponding metadata
            json_key = meta_prefix + filename.replace(".jpg", ".json")
            s3.download_file(bucket, json_key, f"metadata/{filename.replace('.jpg', '.json')}")
            count += 1
        except Exception as e:
            print(f"⚠️ Error downloading {filename}: {e}")

        if max_download and count >= max_download:
            break

    print(f"✅ Downloaded {count} so far...")

    if max_download and count >= max_download:
        break

    if response.get("IsTruncated"):
        continuation_token = response["NextContinuationToken"]
        time.sleep(1)  # polite delay
    else:
        break

print(f"✅ Download complete: {count} image+metadata pairs")
