# -*- coding: utf-8 -*-

import base64
import json
import http.client
from http import HTTPStatus
import os
import uuid
import ssl

class Executor:
    """
    Executor for CLOVA Studio Router API.
    Reads API key from HYPERCLOVA_API_KEY env var if not provided.
    Generates a UUID request ID if not provided.
    """
    def __init__(self, host, api_key=None, request_id=None):
        self._host = host
        key = api_key or os.getenv('HYPERCLOVA_API_KEY')
        if not key:
            raise ValueError("HYPERCLOVA_API_KEY environment variable is not set")
        self._api_key = key if key.startswith('Bearer ') else f"Bearer {key}"
        self._request_id = request_id or str(uuid.uuid4())

    def _send_request(self, request):
        headers = {
            'Content-Type': 'application/json; charset=utf-8',
            'Authorization': self._api_key,
            'X-NCP-CLOVASTUDIO-REQUEST-ID': self._request_id,
        }

        # For local development on macOS, disable SSL verification
        context = ssl._create_unverified_context()
        conn = http.client.HTTPSConnection(self._host, context=context)
        conn.request(
            'POST',
            f"/v1/routers/{os.getenv('CLOVA_ROUTER_ID', 'yl0rl4cd')}/versions/{os.getenv('CLOVA_ROUTER_VERSION', '1')}/route",
            json.dumps(request),
            headers
        )
        response = conn.getresponse()
        status = response.status
        result = json.loads(response.read().decode('utf-8'))
        conn.close()
        return result, status

    def execute(self, request):
        res, status = self._send_request(request)
        if status == HTTPStatus.OK:
            return res['result']
        else:
            raise RuntimeError(f"Router request failed with status {status}: {res}")

if __name__ == '__main__':
    executor = Executor(host='clovastudio.stream.ntruss.com')
    test_request = {
        "query": "미세먼지 정보 알려줘",
        "chatHistory": [
            {"role": "user", "content": "내일 서울의 강수 예보 알려줘"}
        ]
    }
    print(executor.execute(test_request))
