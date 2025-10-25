
import os
from typing import Optional
def send_email_via_smtp(recipient_email: str, subject: str, body: str, attachments: Optional[list]=None) -> tuple[bool, str]:
    """
    Send email using SMTP server configured via env:
    SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, SMTP_USE_TLS (optional)
    If not configured, fall back to printing the email to console (for demo).
    """
    host = os.environ.get("SMTP_HOST")
    port = int(os.environ.get("SMTP_PORT", "0") or 0)
    user = os.environ.get("SMTP_USER")
    pwd = os.environ.get("SMTP_PASS")
    use_tls = os.environ.get("SMTP_USE_TLS", "1") not in ("0","false","False")

    if not host or not port:
        print("SMTP not configured. Printing email instead (demo mode).")
        print("To:", recipient_email)
        print("Subject:", subject)
        print("Body:", body[:1000])
        if attachments:
            print("Attachments:", attachments)
        return True, "Email printed to console (SMTP not configured)"
    try:
        import smtplib
        from email.message import EmailMessage
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = user or f"no-reply@{host}"
        msg["To"] = recipient_email
        msg.set_content(body)
        # attach files
        if attachments:
            for p in attachments:
                try:
                    with open(p, "rb") as f:
                        data = f.read()
                    import mimetypes
                    ctype, _ = mimetypes.guess_type(p)
                    maintype, subtype = (ctype or "application/octet-stream").split("/",1)
                    msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=os.path.basename(p))
                except Exception:
                    pass
        if use_tls:
            server = smtplib.SMTP(host, port, timeout=10)
            server.starttls()
        else:
            server = smtplib.SMTP(host, port, timeout=10)
        if user and pwd:
            server.login(user, pwd)
        server.send_message(msg)
        server.quit()
        return True, "Email sent successfully"
    except Exception as e:
        print("Failed to send email:", e)
        return False, str(e)
