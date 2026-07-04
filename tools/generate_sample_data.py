from __future__ import annotations

import json
from pathlib import Path

from docx import Document
from docx.shared import Pt


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
GOLD = ROOT / "data" / "gold"


DOCUMENTS = {
    "contract_sadr_bridge.txt": """قرارداد شماره ۱۴۰۳-۲۱ شهرداری تهران

موضوع قرارداد: اجرای پروژه بهسازی پل صدر

شهرداری منطقه ۳ تهران اجرای پروژه بهسازی پل صدر را به شرکت راه‌سازان آریا واگذار کرد. این پروژه در محله قیطریه و محدوده بزرگراه صدر قرار دارد.

مهندس ناهید علوی به عنوان ناظر عالی پروژه معرفی شد و مسئولیت کنترل زمان‌بندی، کیفیت اجرا و تأیید صورت‌وضعیت‌ها را بر عهده دارد.

ردیف بودجه عمرانی ۱۴۰۳-۳-۱۷ با مبلغ ۴۲۰ میلیارد ریال، تأمین مالی پروژه بهسازی پل صدر را پوشش می‌دهد.

در متن قرارداد تأکید شده است که شرکت راه‌سازان آریا مجری مستقیم پروژه است و حق واگذاری کامل عملیات اجرایی به پیمانکار فرعی بدون تأیید شهرداری را ندارد.
""",
    "project_report_shariati_widening.txt": """گزارش پیشرفت ماهانه پروژه تعریض خیابان شریعتی

پروژه تعریض خیابان شریعتی در منطقه ۷ تهران و حدفاصل میدان قدس تا خیابان مطهری اجرا می‌شود. هدف پروژه کاهش گره‌های ترافیکی و افزایش ایمنی عبور عابر پیاده است.

شرکت عمران‌گستر پایتخت به عنوان پیمانکار اجرایی پروژه معرفی شده است. مهندس کامران رضایی، مدیر پروژه معاونت فنی و عمرانی، بر اجرای پروژه تعریض خیابان شریعتی نظارت می‌کند.

بودجه مصوب طرح با عنوان ردیف بودجه حمل‌ونقل ۱۴۰۳-۷-۰۹ و اعتبار ۸۷۰ میلیارد ریال برای تأمین مالی پروژه تعریض خیابان شریعتی اختصاص یافته است.

در بازدید میدانی، تأخیر در جابه‌جایی تأسیسات آب و برق ثبت شد، اما رابطه قراردادی پروژه با شرکت عمران‌گستر پایتخت همچنان معتبر است.
""",
    "complaints_137_municipality.txt": """خلاصه گزارش سامانه ۱۳۷

تیکت ۱۳۷-۵۵۸۱: شهروندان محله قیطریه اعلام کردند که پروژه بهسازی پل صدر باعث انسداد پیاده‌رو و افزایش گردوغبار شده است. این شکایت درباره پروژه بهسازی پل صدر ثبت شد.

تیکت ۱۳۷-۵۶۰۲: چند شهروند منطقه ۷ از سر و صدای شبانه ماشین‌آلات در پروژه تعریض خیابان شریعتی شکایت کردند. شکایت به پروژه تعریض خیابان شریعتی ارجاع شد.

تیکت ۱۳۷-۵۶۲۰: شهروندان محله پونک از خرابی روشنایی بوستان نهج‌البلاغه شکایت کردند. این شکایت درباره محله پونک ثبت شد و پروژه مشخصی در گزارش شهروند ذکر نشده است.

تیکت ۱۳۷-۵۶۴۴: ساکنان منطقه ۲ از تأخیر در جمع‌آوری نخاله در اطراف پروژه بازآفرینی میدان صنعت شکایت کردند. موضوع شکایت، پروژه بازآفرینی میدان صنعت است.
""",
    "supervision_minutes.docx": """صورتجلسه نظارت پروژه‌های عمرانی

در جلسه مورخ ۱۴۰۳/۰۸/۱۲، وضعیت سه پروژه عمرانی بررسی شد.

مهندس ناهید علوی تأکید کرد که نظارت عالی پروژه بهسازی پل صدر همچنان بر عهده اوست و گزارش کیفیت آسفالت باید هر هفته ارسال شود.

برای پروژه بازآفرینی میدان صنعت، شرکت شهرسازان نوین به عنوان مجری معرفی شد. این پروژه در منطقه ۲ تهران و محله شهرک غرب اجرا می‌شود.

مهندس سارا ملکی به عنوان ناظر پروژه بازآفرینی میدان صنعت منصوب شد.

ردیف بودجه بازآفرینی ۱۴۰۳-۲-۳۳ با اعتبار ۳۱۵ میلیارد ریال برای تأمین مالی پروژه بازآفرینی میدان صنعت ثبت شد.
""",
    "budget_allocations.txt": """گزارش تخصیص بودجه شهرداری تهران

ردیف بودجه عمرانی ۱۴۰۳-۳-۱۷ با مبلغ ۴۲۰ میلیارد ریال برای تأمین مالی پروژه بهسازی پل صدر تصویب شد.

ردیف بودجه حمل‌ونقل ۱۴۰۳-۷-۰۹ با اعتبار ۸۷۰ میلیارد ریال تأمین مالی پروژه تعریض خیابان شریعتی را بر عهده دارد.

ردیف بودجه بازآفرینی ۱۴۰۳-۲-۳۳ با اعتبار ۳۱۵ میلیارد ریال برای پروژه بازآفرینی میدان صنعت اختصاص یافت.

پروژه بهسازی پل صدر در منطقه ۳ و پروژه تعریض خیابان شریعتی در منطقه ۷ اجرا می‌شوند.
""",
}


EXPECTED_TRIPLETS = [
    ["Contractor", "شرکت راه‌سازان آریا", "EXECUTOR_OF", "Project", "پروژه بهسازی پل صدر"],
    ["Project", "پروژه بهسازی پل صدر", "LOCATED_IN", "Location", "منطقه ۳ تهران"],
    ["Project", "پروژه بهسازی پل صدر", "LOCATED_IN", "Location", "محله قیطریه"],
    ["Official", "مهندس ناهید علوی", "SUPERVISOR_OF", "Project", "پروژه بهسازی پل صدر"],
    ["Budget", "ردیف بودجه عمرانی ۱۴۰۳-۳-۱۷", "FINANCES", "Project", "پروژه بهسازی پل صدر"],
    ["Contractor", "شرکت عمران‌گستر پایتخت", "EXECUTOR_OF", "Project", "پروژه تعریض خیابان شریعتی"],
    ["Project", "پروژه تعریض خیابان شریعتی", "LOCATED_IN", "Location", "منطقه ۷ تهران"],
    ["Official", "مهندس کامران رضایی", "SUPERVISOR_OF", "Project", "پروژه تعریض خیابان شریعتی"],
    ["Budget", "ردیف بودجه حمل‌ونقل ۱۴۰۳-۷-۰۹", "FINANCES", "Project", "پروژه تعریض خیابان شریعتی"],
    ["Complaint", "تیکت ۱۳۷-۵۵۸۱", "COMPLAINS_ABOUT", "Project", "پروژه بهسازی پل صدر"],
    ["Complaint", "تیکت ۱۳۷-۵۶۰۲", "COMPLAINS_ABOUT", "Project", "پروژه تعریض خیابان شریعتی"],
    ["Complaint", "تیکت ۱۳۷-۵۶۲۰", "COMPLAINS_ABOUT", "Location", "محله پونک"],
    ["Complaint", "تیکت ۱۳۷-۵۶۴۴", "COMPLAINS_ABOUT", "Project", "پروژه بازآفرینی میدان صنعت"],
    ["Contractor", "شرکت شهرسازان نوین", "EXECUTOR_OF", "Project", "پروژه بازآفرینی میدان صنعت"],
    ["Project", "پروژه بازآفرینی میدان صنعت", "LOCATED_IN", "Location", "منطقه ۲ تهران"],
    ["Project", "پروژه بازآفرینی میدان صنعت", "LOCATED_IN", "Location", "محله شهرک غرب"],
    ["Official", "مهندس سارا ملکی", "SUPERVISOR_OF", "Project", "پروژه بازآفرینی میدان صنعت"],
    ["Budget", "ردیف بودجه بازآفرینی ۱۴۰۳-۲-۳۳", "FINANCES", "Project", "پروژه بازآفرینی میدان صنعت"],
]


def write_txt(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def write_docx(path: Path, text: str) -> None:
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Tahoma"
    style.font.size = Pt(11)
    for block in text.split("\n\n"):
        para = doc.add_paragraph(block)
        para.paragraph_format.right_to_left = True
    doc.save(path)


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    GOLD.mkdir(parents=True, exist_ok=True)
    for old in RAW.iterdir():
        if old.suffix.lower() in {".txt", ".docx", ".pdf"}:
            old.unlink()

    for name, text in DOCUMENTS.items():
        path = RAW / name
        if path.suffix == ".txt":
            write_txt(path, text)
        elif path.suffix == ".docx":
            write_docx(path, text)

    (GOLD / "expected_triplets.json").write_text(
        json.dumps(
            {
                "description": "Expected ontology facts for sample municipality documents.",
                "triplets": [
                    {
                        "subject_type": s_type,
                        "subject_name": s_name,
                        "relation": rel,
                        "object_type": o_type,
                        "object_name": o_name,
                    }
                    for s_type, s_name, rel, o_type, o_name in EXPECTED_TRIPLETS
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"created {len(DOCUMENTS)} raw files")
    print(f"created {len(EXPECTED_TRIPLETS)} expected triplets")


if __name__ == "__main__":
    main()
