import smtplib
from email.message import EmailMessage

from app.core.config import settings


class EmailService:
    @staticmethod
    def send_verification_email(to_email: str, token: str):
        verify_url = f"{settings.frontend_url}/verify-email?token={token}"

        msg = EmailMessage()
        msg["Subject"] = "Verify your NeuroXAI account"
        msg["From"] = settings.smtp_username
        msg["To"] = to_email
        msg.set_content(
            f"""
Welcome to NeuroXAI.

Please verify your email by clicking the link below:

{verify_url}

This link will expire in {settings.verification_token_expire_hours} hours.
""".strip()
        )

        with smtplib.SMTP(settings.smtp_host, settings.smtp_port) as server:
            server.starttls()
            server.login(settings.smtp_username, settings.smtp_password)
            server.send_message(msg)