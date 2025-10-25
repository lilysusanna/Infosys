import os
import base64
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import Optional, List
import streamlit as st

try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    OAUTH_AVAILABLE = True
except ImportError:
    OAUTH_AVAILABLE = False

# Gmail API scopes
SCOPES = ['https://www.googleapis.com/auth/gmail.send']

def get_gmail_service():
    """Get authenticated Gmail service using OAuth2."""
    if not OAUTH_AVAILABLE:
        raise ImportError("Google OAuth libraries not installed. Run: pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client")
    
    creds = None
    token_file = 'token.json'
    
    # Load existing credentials
    if os.path.exists(token_file):
        creds = Credentials.from_authorized_user_file(token_file, SCOPES)
    
    # If no valid credentials, get new ones
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # Check if credentials.json exists
            if not os.path.exists('credentials.json'):
                raise FileNotFoundError(
                    "credentials.json not found. Please download it from Google Cloud Console:\n"
                    "1. Go to https://console.cloud.google.com/\n"
                    "2. Create a new project or select existing\n"
                    "3. Enable Gmail API\n"
                    "4. Create OAuth 2.0 credentials\n"
                    "5. Download as credentials.json"
                )
            
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save credentials for next run
        with open(token_file, 'w') as token:
            token.write(creds.to_json())
    
    return build('gmail', 'v1', credentials=creds)

def send_email_oauth(recipient_email: str, subject: str, body: str, attachments: Optional[List[str]] = None) -> tuple[bool, str]:
    """
    Send email using Gmail API with OAuth2.
    
    Args:
        recipient_email: Email address to send to
        subject: Email subject
        body: Email body text
        attachments: List of file paths to attach
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        if not OAUTH_AVAILABLE:
            return False, "OAuth libraries not installed. Run: pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client"
        
        service = get_gmail_service()
        
        # Create message
        message = MIMEMultipart()
        message['to'] = recipient_email
        message['subject'] = subject
        
        # Add body
        message.attach(MIMEText(body, 'plain'))
        
        # Add attachments
        if attachments:
            for file_path in attachments:
                if os.path.exists(file_path):
                    with open(file_path, 'rb') as attachment:
                        part = MIMEBase('application', 'octet-stream')
                        part.set_payload(attachment.read())
                    
                    encoders.encode_base64(part)
                    part.add_header(
                        'Content-Disposition',
                        f'attachment; filename= {os.path.basename(file_path)}'
                    )
                    message.attach(part)
        
        # Encode message
        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')
        
        # Send message
        send_message = service.users().messages().send(
            userId='me',
            body={'raw': raw_message}
        ).execute()
        
        return True, f"Email sent successfully! Message ID: {send_message['id']}"
        
    except FileNotFoundError as e:
        return False, f"Configuration error: {str(e)}"
    except HttpError as e:
        return False, f"Gmail API error: {str(e)}"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"

def setup_oauth_instructions():
    """Display OAuth2 setup instructions in Streamlit."""
    st.markdown("### 🔐 OAuth2 Email Setup Instructions")
    
    st.markdown("""
    **Step 1: Enable Gmail API**
    1. Go to [Google Cloud Console](https://console.cloud.google.com/)
    2. Create a new project or select an existing one
    3. Enable the Gmail API
    4. Go to "Credentials" → "Create Credentials" → "OAuth 2.0 Client IDs"
    5. Choose "Desktop application"
    6. Download the credentials as `credentials.json`
    7. Place `credentials.json` in your project folder
    
    **Step 2: Install Required Libraries**
    ```bash
    pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client
    ```
    
    **Step 3: First Run**
    - The first time you send an email, your browser will open for authentication
    - Grant permissions to the app
    - A `token.json` file will be created for future use
    """)
    
    # Check if credentials.json exists
    if os.path.exists('credentials.json'):
        st.success("✅ credentials.json found!")
    else:
        st.warning("⚠️ credentials.json not found. Please follow Step 1 above.")
    
    # Check if required libraries are installed
    if OAUTH_AVAILABLE:
        st.success("✅ OAuth libraries are installed!")
    else:
        st.error("❌ OAuth libraries not installed. Please run: pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client")

def test_oauth_connection():
    """Test OAuth2 connection and display status."""
    try:
        if not OAUTH_AVAILABLE:
            return False, "OAuth libraries not installed"
        
        service = get_gmail_service()
        # Try to get user profile to test connection
        profile = service.users().getProfile(userId='me').execute()
        return True, f"Connected as: {profile.get('emailAddress', 'Unknown')}"
    except Exception as e:
        return False, f"Connection failed: {str(e)}"
