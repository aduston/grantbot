import os
import tempfile
import google.auth
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload


def upload_markdown_to_gdoc(markdown: str, output_file_name: str) -> str:
    """Insert new file.
    Returns : Id's of the file uploaded

    Load pre-authorized user credentials from the environment.
    What I had to do:

    gcloud auth application-default login \
        --scopes=openid,https://www.googleapis.com/auth/userinfo.email,\
            https://www.googleapis.com/auth/cloud-platform,\
                https://www.googleapis.com/auth/drive.file
    gcloud auth application-default set-quota-project aad-personal
    """
    creds, _ = google.auth.default()
    service = build('drive', 'v3', credentials=creds)
    with tempfile.NamedTemporaryFile(delete=False) as fp:
        fp.write(markdown.encode('utf-8'))
        input_file_name = fp.name
        fp.close()
        file_metadata = {
            'name': output_file_name,
            'mimeType': 'application/vnd.google-apps.document'
        }
        media = MediaFileUpload(input_file_name, mimetype='text/markdown')
        # pylint: disable=E1101
        file = service.files().create(body=file_metadata,
                                      media_body=media,
                                      fields='id').execute()
    return file.get('id')


if __name__ == '__main__':
    with open(
            os.path.expanduser("~/code/grantbot/grantbot/sample_report.md"),
            "r", encoding="utf-8") as f:
        file_markdown = f.read()
    upload_markdown_to_gdoc(file_markdown, "Sample Report")
