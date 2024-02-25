import os
from twilio.rest import Client

# Your Twilio account SID and auth token
account_sid = os.environ['TWILIO_ACCOUNT_SID']
auth_token = os.environ['TWILIO_AUTH_TOKEN']

client = Client(account_sid, auth_token)

# Send an SMS
message = client.messages.create(
    body='uh oh looks like someone fell! Would you like to send help?',
    from_=os.environ['SENDER_PHONE_NUMBER'],
    to=os.enviorn['RECIPIENT_PHONE_NUMBER']
)

# Check for a response
messages = client.messages.list(from_=os.environ['RECIPIENT_PHONE_NUMBER'])
for record in messages:
    if record.body.lower() == 'yes':
        # If the response is 'yes', send another message
        message = client.messages.create(
            body='Okay, help is on the way!',
            from_=os.environ['SENDER_PHONE_NUMBER'],
            to=os.environ['RECIPIENT_PHONE_NUMBER']
        )
        break

