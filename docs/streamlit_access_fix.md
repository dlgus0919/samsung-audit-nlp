# Streamlit 접근 오류 복구 런북

증상:
- 외부 PC 또는 비로그인 상태에서 앱 접속 시 아래 메시지 노출
- `You do not have access to this app or it does not exist`

대상 앱:
- `https://samsung-audit-nlp-8o7p9ncsmswujv2pjjc9rt.streamlit.app/`

## 1) 공유 설정 확인 (최우선)

1. 앱 소유자 계정으로 [share.streamlit.io](https://share.streamlit.io) 로그인
2. 대상 앱 선택
3. 앱 설정에서 공유 범위를 `Public`(anyone can view)으로 변경
4. 로그아웃 브라우저/시크릿 창에서 URL 재검증

정상 결과:
- 로그인 없이 랜딩 및 앱 화면 표시

## 2) 앱 존재/배포 상태 점검

1. 앱 URL이 현재 배포와 동일한지 확인 (재생성으로 URL이 바뀐 경우가 있음)
2. 앱 빌드 로그에서 실패 여부 확인
3. 실패 시 `Reboot app` 또는 최근 정상 커밋으로 재배포

## 3) 권한/워크스페이스 이슈 점검

1. 개인 워크스페이스가 아닌 조직 워크스페이스에 묶인 앱인지 확인
2. 조직 정책으로 비공개 강제되는지 확인
3. 필요 시 공개 가능한 워크스페이스로 앱 이전

## 4) 미해결 시 지원 요청

아래 정보를 첨부하여 Streamlit 지원에 케이스 접수:
- 앱 URL
- 발생 시간(타임존 포함)
- 증상 스크린샷
- 앱 소유자 계정/워크스페이스 정보
- 최근 배포 커밋 SHA

## 참고 링크

- [Streamlit Community Cloud 관리 문서](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app)
- [Deployment issues knowledge base](https://docs.streamlit.io/knowledge-base/deploy)
- [유사 오류 토론 스레드](https://discuss.streamlit.io/t/you-do-not-have-access-to-this-app-or-it-does-not-exist-error/120257)
