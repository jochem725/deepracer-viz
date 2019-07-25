import requests
from bs4 import BeautifulSoup
import cv2
import numpy as np

with requests.Session() as s:
    URL = "https://192.168.1.101/"
    post_login_url = "https://192.168.1.101/login"
    video_url = "https://192.168.1.101/route?topic=/video_mjpeg&width=480&height=360"

    # Get the CSRF Token
    response = s.get(URL, verify=False)
    soup = BeautifulSoup(response.text, 'lxml')
    csrf_token = soup.select_one('meta[name="csrf-token"]')['content']
    headers = {'X-CSRFToken': csrf_token}
    # print("CSRF token found: " + str(csrf_token))

    # Login to the DeepRacer web interface with Post
    payload = {'password': 'uGRqirr3'}
    post = s.post(post_login_url, data=payload, headers=headers, verify=False)
    # print("Login: " + post.text)

    # Get the video stream
    video_stream = s.get(video_url, stream=True, verify=False)
    if video_stream.status_code == 200:
        print("Video Connected!")
        bytes = bytes()
        for chunk in video_stream.iter_content(chunk_size=1024):
            bytes += chunk
            a = bytes.find(b'\xff\xd8')  # Marker byte pair
            b = bytes.find(b'\xff\xd9')  # Trailing byte pair
            #  If both byte pairs on in the stream then build the jpeg
            if a != -1 and b != -1:
                jpg = bytes[a:b + 2]
                bytes = bytes[b + 2:]
                i = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                cv2.imshow('Raw Image', i)
                if cv2.waitKey(1) == 27:
                    exit(0)
    else:
        print("Received unexpected status code {}".format(video_stream.status_code))
