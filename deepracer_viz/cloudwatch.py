import boto3


def list_log_streams(log_group, limit=20):
    client = boto3.client('logs')

    kwargs = {
        'logGroupName': log_group,
        'descending': True,
        'orderBy': 'LastEventTime',
        'limit': limit
    }

    logs = client.describe_log_streams(**kwargs)

    return list(map(lambda x: x['logStreamName'], logs['logStreams']))


class CloudWatchLogs():
    def __init__(self, log_group, stream_name):
        self.startTime = 0
        self.log_group = log_group
        self.stream_name = stream_name

    def get_log_events(self):
        client = boto3.client('logs')

        kwargs = {
            'logGroupName': self.log_group,
            'logStreamNames': [self.stream_name],
            'startTime': self.startTime,
            'endTime': 2000000000000,  # Some date far in the future.
            'limit': 10000
        }

        while True:
            response = client.filter_log_events(**kwargs)
            self.startTime = response['events'][-1]['timestamp']

            yield from response['events']

            if 'nextToken' in response:
                kwargs['nextToken'] = response['nextToken']
            else:
                break
