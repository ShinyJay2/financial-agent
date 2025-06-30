# Financial AI Agent README

이 문서는 **Financial AI Agent** 프로젝트의 초기 설정, 실행, 테스트 방법을 상세히 안내합니다.

---

## 1. 프로젝트 클론 및 이동

```bash
# 터미널을 열고 작업 디렉토리로 이동
cd ~/Projects

# GitHub에서 레포지토리 클론
git clone https://github.com/ShinyJay2/financial-agent.git

# 프로젝트 디렉토리로 이동
cd financial-agent
```

## 2. 가상환경 생성 및 활성화

```bash
# Python 3로 가상환경 생성 (이름: finagent)
python3 -m venv finagent

# 가상환경 활성화
source finagent/bin/activate
```

> **참고:** 활성화되면 프롬프트 앞에 `(finagent)`가 표시됩니다.

## 3. 의존성 설치

```bash
# pip 최신 버전으로 업데이트
pip install --upgrade pip

# requirements.txt에 명시된 패키지 설치
pip install -r requirements.txt
```

## 4. 환경 변수 설정

```bash
# .env.example을 .env로 복사
cp .env.example .env

# .env 파일을 열어 API 키 및 설정 입력
# 편집기 예시: VS Code 사용 시
code .env
```

.env 파일 예시:

```dotenv
HYPERCLOVA_API_KEY=nv-여기에-실제-키-입력
HYPERCLOVASTUDIO_REQUEST_ID=실제-REQUEST-ID-입력
# DATABASE_URL은 사용 시에만 입력
DATABASE_URL=
```

> **주의:** `.env` 파일은 `.gitignore`에 포함되어 있어 Git에 커밋되지 않습니다.

## 5. 서버 실행

```bash
# FastAPI 서버 실행 (코드 변경 시 자동 재시작)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

* **--reload**: 소스 코드 변경 시 자동으로 서버를 재시작합니다.
* **--host 0.0.0.0**: 모든 네트워크 인터페이스에서 접근 가능하도록 설정합니다.
* **--port 8000**: 포트 8000번으로 서비스합니다.

실행 후 터미널에 아래 메시지가 출력되면 정상 실행 중입니다:

```
INFO: Will watch for changes in these directories: ['.../financial-agent']
INFO: Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

## 6. API 테스트 방법

### 6.1 Swagger UI (웹 UI)

1. 웹 브라우저에서 다음 URL 접속:

   ```
   http://127.0.0.1:8000/docs
   ```
2. `GET /agent` 항목의 **Try it out** 클릭
3. **question** 필드에 질문 입력 (예: `안녕하세요`)
4. **Execute** 버튼 클릭 → 결과 JSON 확인

### 6.2 curl 명령어

```bash
curl -G "http://127.0.0.1:8000/agent" \
     --data-urlencode "question=안녕하세요" \
     -H "Authorization: Bearer ${HYPERCLOVA_API_KEY}" \
     -H "X-NCP-CLOVASTUDIO-REQUEST-ID: ${HYPERCLOVASTUDIO_REQUEST_ID}"
```

* `-G` 옵션: GET 방식으로 쿼리 파라미터를 전송
* `--data-urlencode`: 질문에 포함된 한글을 URL 인코딩하여 안전하게 전송

성공 시 예시 응답:

```json
{"answer":"이 부분은 모의 응답입니다."}
```

> **모드핑크(response mocking)**
> 환경 변수 `HYPERCLOVA_API_KEY`가 `dummy`로 시작할 경우, 실제 API 호출 대신 아래 모의 응답을 반환합니다:
>
> ```text
> 이 부분은 모의 응답입니다.
> ```

---

## 7. 프로젝트 구조

```
financial-agent/
├── .gitignore           # Git 제외 대상
├── .env.example         # 환경 변수 템플릿
├── README.md            # 이 가이드 문서
├── requirements.txt     # Python 패키지 목록
├── app/                 # 애플리케이션 코드
│   ├── main.py          # FastAPI 메인 진입점
│   ├── config.py        # 환경 변수 설정
│   ├── hyperclova_client.py  # LLM 호출 모듈
│   └── yahoo_client.py  # yfinance 데이터 모듈
├── tests/               # 테스트 코드
│   └── test_main.py     # 기본 엔드포인트 테스트
└── finagent/            # Python 가상환경 (Git 무시)
```

---

## 8. 이후 작업 계획

1. **질의 핸들러 구현**: 가격 분석, 거래량 상위, 이동평균, 조건 필터링 등
2. **데이터 소스 확장**: Alpha Vantage 등 대량 데이터 API 연동
3. **DB 연동**: PostgreSQL 및 Alembic 마이그레이션
4. **고급 RAG**: 벡터 검색, 하이브리드 검색 등 통합
5. **배포 자동화**: NaverCloud VM에 Systemd 서비스 구성

---

If any questions arise, feel free to ask.
