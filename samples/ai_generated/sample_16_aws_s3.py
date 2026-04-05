"""
AWS S3操作ユーティリティ
Python基礎: 例外処理、コンテキストマネージャ、パス操作
DevOps語録: boto3, s3, upload, download, bucket
"""
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class S3Client:
    """S3操作クライアント"""

    def __init__(self, region_name: str = "ap-northeast-1"):
        import boto3
        from botocore.exceptions import ClientError
        self.s3 = boto3.client("s3", region_name=region_name)
        self.ClientError = ClientError

    def upload_file(self, local_path: Path, bucket: str, key: str) -> bool:
        """ファイルをS3にアップロードする"""
        try:
            self.s3.upload_file(str(local_path), bucket, key)
            logger.info(f"アップロード完了: s3://{bucket}/{key}")
            return True
        except self.ClientError as e:
            logger.error(f"アップロード失敗: {e}")
            return False

    def download_file(self, bucket: str, key: str, local_path: Path) -> bool:
        """S3からファイルをダウンロードする"""
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            self.s3.download_file(bucket, key, str(local_path))
            logger.info(f"ダウンロード完了: {local_path}")
            return True
        except self.ClientError as e:
            logger.error(f"ダウンロード失敗: {e}")
            return False

    def list_objects(self, bucket: str, prefix: str = "", max_keys: int = 1000) -> list[dict]:
        """オブジェクト一覧を取得する"""
        try:
            response = self.s3.list_objects_v2(
                Bucket=bucket,
                Prefix=prefix,
                MaxKeys=max_keys,
            )
            objects = response.get("Contents", [])
            return [
                {
                    "key": obj["Key"],
                    "size": obj["Size"],
                    "last_modified": str(obj["LastModified"]),
                }
                for obj in objects
            ]
        except self.ClientError as e:
            logger.error(f"一覧取得失敗: {e}")
            return []

    def delete_object(self, bucket: str, key: str) -> bool:
        """オブジェクトを削除する"""
        try:
            self.s3.delete_object(Bucket=bucket, Key=key)
            logger.info(f"削除完了: s3://{bucket}/{key}")
            return True
        except self.ClientError as e:
            logger.error(f"削除失敗: {e}")
            return False

    def put_json(self, bucket: str, key: str, data: dict) -> bool:
        """JSONデータをS3に保存する"""
        try:
            body = json.dumps(data, ensure_ascii=False, indent=2)
            self.s3.put_object(
                Bucket=bucket,
                Key=key,
                Body=body.encode("utf-8"),
                ContentType="application/json",
            )
            logger.info(f"JSON保存完了: s3://{bucket}/{key}")
            return True
        except self.ClientError as e:
            logger.error(f"JSON保存失敗: {e}")
            return False

    def get_json(self, bucket: str, key: str) -> Optional[dict]:
        """S3からJSONデータを読み込む"""
        try:
            response = self.s3.get_object(Bucket=bucket, Key=key)
            body = response["Body"].read().decode("utf-8")
            return json.loads(body)
        except self.ClientError as e:
            logger.error(f"JSON読み込み失敗: {e}")
            return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print("S3クライアント - boto3が必要です")
