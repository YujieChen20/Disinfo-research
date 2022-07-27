# adapted from / reference:
# ... https://developers.google.com/calendar/quickstart/python
# ... https://github.com/googleworkspace/python-samples/blob/master/calendar/quickstart/quickstart.py
# ... https://bitbucket.org/kingmray/django-google-calendar/src/master/calendar_api/calendar_api.py

# see docs:
# ... https://developers.google.com/calendar/overview
# ... https://developers.google.com/calendar/v3/reference/calendars

import os
import json
from pprint import pprint
from datetime import datetime
# import datetime
# import pytz

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
# from oauth2client.service_account import ServiceAccountCredentials # deprecated, causing issues
from google.oauth2.service_account import Credentials

GOOGLE_API_CREDENTIALS = os.getenv("GOOGLE_API_CREDENTIALS")

# see: # https://developers.google.com/identity/protocols/oauth2/scopes#drive
SCOPES = [
    "https://www.googleapis.com/auth/drive"  # See, edit, create, and delete all of your Google Drive files
    #"https://www.googleapis.com/auth/drive.appdata",   # See, create, and delete its own configuration data in your Google Drive
    #"https://www.googleapis.com/auth/drive.file",   # See, edit, create, and delete only the specific Google Drive files you use with this app
    #https://www.googleapis.com/auth/drive.metadata,   # View and manage metadata of files in your Google Drive
    #"https://www.googleapis.com/auth/ drive.metadata.readonly",   # See information about your Google Drive files
    #"https://www.googleapis.com/auth/drive.photos.readonly",   # View the photos, videos and albums in your Google Photos
    #"https://www.googleapis.com/auth/drive.readonly",   # See and download all your Google Drive files
    #"https://www.googleapis.com/auth/drive.scripts",   # Modify your Google Apps Script scripts' behavior
]


class GoogleDriveService:

    def __init__(self, credentials=None):
        self.credentials = credentials
        if not self.credentials:
            GOOGLE_CREDS_JSON = json.loads(GOOGLE_API_CREDENTIALS)
            # self.credentials = ServiceAccountCredentials._from_parsed_json_keyfile(CREDS_JSON, SCOPES)
            # ...
            # oauthclient2 is deprecated, so use google-auth instead:
            # ... https://google-auth.readthedocs.io/en/master/user-guide.html
            #
            #self.credentials = Credentials.from_service_account_file(GOOGLE_API_CREDENTIALS).with_scopes(SCOPES)
            self.credentials = Credentials.from_service_account_info(GOOGLE_CREDS_JSON).with_scopes(SCOPES)

        self.client = build("drive", "v3", credentials=self.credentials)

    def list_files(self):
        results = self.client.files().list(pageSize=10, fields="nextPageToken, files(id, name)").execute()
        items = results.get('files', [])
        if not items:
            print('No files found.')
            return
        print('Files:')
        for item in items:
            print(u'{0} ({1})'.format(item['name'], item['id']))
        return results

    def get_file(self, file_id):
        # https://developers.google.com/drive/api/v3/reference/files/get
        return self.client.files().get(fileId=file_id).execute()


if __name__ == "__main__":
    gcal_service = GoogleDriveService()
    cals = gcal_service.list_files()
    pprint(cals)
