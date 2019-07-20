import boto3
import datetime


class KinesisVideoStream():

    def __init__(self, name: str):
        self.client = boto3.client('kinesisvideo')

        self.streamName = name
        self.endpoint = self.client.get_data_endpoint(
            StreamName=self.streamName,
            APIName='GET_HLS_STREAMING_SESSION_URL'
        )['DataEndpoint']

    def get_live_streaming_session_url(self):
        video_client = boto3.client(
            "kinesis-video-archived-media", endpoint_url=self.endpoint)

        return video_client.get_hls_streaming_session_url(
            StreamName=self.streamName,
            PlaybackMode="LIVE"
        )['HLSStreamingSessionURL']

    def get_time_range_url(self, startDate: datetime, endDate: datetime):
        video_client = boto3.client(
            "kinesis-video-archived-media", endpoint_url=self.endpoint)

        return video_client.get_hls_streaming_session_url(
            StreamName=self.streamName,
            PlaybackMode="ON_DEMAND",
            HLSFragmentSelector={
                'FragmentSelectorType': 'SERVER_TIMESTAMP',
                'TimestampRange': {
                    'StartTimestamp': startDate,
                    'EndTimestamp': endDate
                }
            },
        )['HLSStreamingSessionURL']
