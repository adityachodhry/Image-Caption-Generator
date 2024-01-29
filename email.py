import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication

# Sender and recipient email addresses
sender_email = "aditya.choudhary@retvenstechnologies.com"
recipient_email = "kumar222107@gmail.com"

# Your Gmail credentials
username = "aditya.choudhary@retvenstechnologies.com"
password = "Ac@bhul.bhai"

# Create the email message
subject = "For Check"
body = "Kya haal hai bhai"

message = MIMEMultipart()
message["From"] = sender_email
message["To"] = recipient_email
message["Subject"] = subject
message.attach(MIMEText(body, "plain"))

# Attach Excel file
excel_filename = "Ammenities.xlsx"
with open(excel_filename, "rb") as excel_file:
    excel_attachment = MIMEApplication(excel_file.read(), _subtype="xlsx")
    excel_attachment.add_header("Content-Disposition", f"attachment; filename={excel_filename}")
    message.attach(excel_attachment)


# Connect to Gmail's SMTP server
with smtplib.SMTP("smtp.gmail.com", 587) as server:
    # Start the TLS connection
    server.starttls()

    # Login to your Gmail account
    server.login(username, password)

    # Send the email
    server.sendmail(sender_email, recipient_email, message.as_string())
