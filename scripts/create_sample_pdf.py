"""
테스트용 샘플 PDF 생성 스크립트

테스트에 사용된 청크 데이터를 PDF 문서로 변환합니다.
"""

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from pathlib import Path


# 샘플 문서 내용 (테스트 청크 통합)
DOCUMENT_CONTENT = """
구매요청 승인 프로세스 규정

제1장 총칙

제1조 (목적)
이 규정은 회사의 구매요청 승인 프로세스에 관한 사항을 정함을 목적으로 한다.
구매요청 승인 프로세스는 부서별 구매 수요를 체계적으로 관리하고, 
예산 범위 내에서 효율적인 자원 배분을 실현하기 위한 업무 절차이다.

제2조 (적용범위)
이 규정은 본사 및 지사의 모든 부서에서 발생하는 구매요청에 적용된다.

제3조 (용어의 정의)
1. "구매요청서"란 물품 또는 서비스의 구매를 요청하는 공식 문서를 말한다.
2. "승인권자"란 구매요청을 검토하고 승인할 권한이 있는 자를 말한다.
3. "구매담당자"란 승인된 구매요청을 실제로 처리하는 담당자를 말한다.

제4조 (구매요청서 작성)
구매요청자는 구매가 필요한 경우 전산시스템을 통해 구매요청서를 작성한다.


제2장 프로세스 단계

제6조 (구매요청 승인 프로세스 단계)
구매요청 승인 프로세스는 다음의 단계로 진행된다.

1단계: 구매요청서 접수
구매담당자는 전자결재 시스템을 통해 접수된 구매요청서를 확인한다.
접수된 요청서는 요청번호를 부여받으며, 구매담당자는 형식적 요건을 검토한다.

2단계: 예산 확인
재무담당자는 해당 구매요청에 대한 예산 가용 여부를 확인한다.
예산이 부족한 경우 예산조정 요청을 진행하거나 구매요청을 반려할 수 있다.

3단계: 규격 검토
기술담당자는 요청된 품목의 기술적 사양과 규격이 적정한지 검토한다.
필요시 대체 품목이나 수정된 사양을 제안할 수 있다.


제3장 조직 및 역할

별첨: 조직도 및 역할 정의

주요 역할

구매팀장
- 구매요청 승인 프로세스의 총괄 책임자
- 100만원 이상의 구매요청에 대한 최종 승인 권한 보유
- 공급업체 계약 체결 권한

구매담당자
- 구매요청서 접수 및 형식 검토 수행
- 견적서 수집 및 비교분석 담당
- 발주서 작성 및 발송 처리
- 구매요청 승인 프로세스의 실무 담당자

재무담당자
- 구매요청에 대한 예산 가용 여부 확인
- 예산 초과 시 예산조정 요청 프로세스 진행
- 구매 대금 지급 승인

기술담당자
- 기술 사양 및 규격 검토
- 대체 품목 또는 사양 변경 제안
- 납품 물품의 기술적 검수


제4장 발주 및 검수

제12조 (발주 처리)
구매요청 승인 프로세스를 통해 승인된 건에 대하여 구매담당자는 다음을 수행한다.

4단계: 견적서 수집
구매담당자는 3개 이상의 공급업체로부터 견적서를 수집한다.
견적서에는 품목, 단가, 납기, 결제조건이 포함되어야 한다.

5단계: 업체 선정
수집된 견적서를 비교하여 최적의 공급업체를 선정한다.
100만원 이상의 경우 구매팀장의 승인을 받아야 한다.

6단계: 발주서 발송
선정된 업체에 발주서를 발송한다.
발주서에는 품목, 수량, 단가, 납품일, 결제조건을 명시한다.

제13조 (입고 검수)
7단계: 입고 검수
납품된 물품에 대해 구매담당자와 기술담당자가 공동으로 검수를 진행한다.

제14조 (대금 지급)
8단계: 대금 지급
검수 완료 후 재무담당자는 공급업체에 대금을 지급한다.
대금 지급은 구매요청 승인 프로세스의 최종 단계이다.


제5장 분기 조건

제15조 (승인 분기 조건)
구매요청 승인 프로세스에서 다음의 조건에 따라 분기가 발생한다.

승인/반려 분기 (XOR Gateway)
- 예산 확인 결과 예산이 충분한 경우: 규격 검토 단계로 진행
- 예산 확인 결과 예산이 부족한 경우: 예산조정 요청 또는 반려

금액별 승인권자 분기 (XOR Gateway)  
- 구매금액이 50만원 미만인 경우: 구매담당자 승인
- 구매금액이 50만원 이상 100만원 미만인 경우: 구매팀장 승인
- 구매금액이 100만원 이상인 경우: 임원 승인

제16조 (검수 분기)
검수 결과에 따른 분기 (XOR Gateway)
- 검수 합격인 경우: 대금 지급 단계로 진행
- 검수 불합격인 경우: 반품 처리 및 재발주 검토


부칙

제1조 (시행일)
이 규정은 2024년 1월 1일부터 시행한다.

제2조 (경과조치)
이 규정 시행 전에 진행 중인 구매요청 건은 종전의 규정에 따른다.
"""


def create_sample_pdf():
    """샘플 PDF 생성 (ReportLab 사용)"""
    
    # 출력 경로
    output_dir = Path(__file__).parent.parent / "uploads"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "purchase_approval_process.pdf"
    
    print(f"📄 PDF 생성 중...")
    
    # 한글 폰트 등록
    pdfmetrics.registerFont(UnicodeCIDFont('HYGothic-Medium'))
    
    # 문서 생성
    doc = SimpleDocTemplate(
        str(output_file),
        pagesize=A4,
        leftMargin=20*mm,
        rightMargin=20*mm,
        topMargin=20*mm,
        bottomMargin=20*mm
    )
    
    # 스타일 정의
    styles = getSampleStyleSheet()
    
    # 한글 스타일
    title_style = ParagraphStyle(
        'KoreanTitle',
        parent=styles['Title'],
        fontName='HYGothic-Medium',
        fontSize=18,
        spaceAfter=20
    )
    
    heading_style = ParagraphStyle(
        'KoreanHeading',
        parent=styles['Heading1'],
        fontName='HYGothic-Medium',
        fontSize=14,
        spaceBefore=15,
        spaceAfter=10
    )
    
    subheading_style = ParagraphStyle(
        'KoreanSubheading',
        parent=styles['Heading2'],
        fontName='HYGothic-Medium',
        fontSize=12,
        spaceBefore=10,
        spaceAfter=5
    )
    
    body_style = ParagraphStyle(
        'KoreanBody',
        parent=styles['Normal'],
        fontName='HYGothic-Medium',
        fontSize=10,
        leading=14,
        spaceBefore=3,
        spaceAfter=3
    )
    
    bullet_style = ParagraphStyle(
        'KoreanBullet',
        parent=body_style,
        leftIndent=15,
        bulletIndent=5
    )
    
    # 문서 내용 구성
    story = []
    
    for line in DOCUMENT_CONTENT.strip().split('\n'):
        line = line.strip()
        
        if not line:
            story.append(Spacer(1, 5*mm))
            continue
        
        # 스타일 결정
        if line.startswith("구매요청 승인 프로세스 규정"):
            story.append(Paragraph(line, title_style))
        elif line.startswith("제") and "장" in line and len(line) < 20:
            story.append(Paragraph(line, heading_style))
        elif line.startswith("제") and "조" in line:
            story.append(Paragraph(line, subheading_style))
        elif "단계:" in line or line.endswith("단계"):
            story.append(Paragraph(f"<b>{line}</b>", body_style))
        elif line.startswith("-"):
            story.append(Paragraph(f"• {line[1:].strip()}", bullet_style))
        elif line.startswith("별첨") or line.startswith("주요 역할") or line.startswith("부칙"):
            story.append(Paragraph(f"<b>{line}</b>", subheading_style))
        elif line in ["구매팀장", "구매담당자", "재무담당자", "기술담당자"]:
            story.append(Spacer(1, 3*mm))
            story.append(Paragraph(f"<b>{line}</b>", body_style))
        else:
            story.append(Paragraph(line, body_style))
    
    # PDF 생성
    doc.build(story)
    
    print(f"✅ PDF 저장 완료: {output_file}")
    print(f"   파일 크기: {output_file.stat().st_size:,} bytes")
    
    return output_file


if __name__ == "__main__":
    create_sample_pdf()
